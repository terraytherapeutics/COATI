"""
Causal and non-causal transformer blocks. 
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F


class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, x):
        return (
            0.5
            * x
            * (
                1.0
                + torch.tanh(
                    math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))
                )
            )
        )


class RotaryEmbedding(torch.nn.Module):
    def __init__(
        self,
        n_seq=256,
        n_embd=128,
        n_tok=512,
        n_head=8,
        norm_embed=False,
        device=torch.device("cpu"),
        dtype=torch.float,
        base=10000,
    ):
        """
        Eq. (34) of https://arxiv.org/pdf/2104.09864.pdf
        also inspired by https://blog.eleuther.ai/rotary-embeddings/
        The rotation is done after the hidden dimension is split into heads.
        so, the cached sin/cos tensors operate on a space (n_embd // n_head)

        Args:
            n_seq: Maximum sequence dimension.
            n_embd: embedding dimension (pre head split)
            n_tok: size of tokenspace.
            n_head: number of attention heads.
        """
        super().__init__()
        assert n_embd % (2 * n_head) == 0
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, (n_embd // n_head), 2, device=device).float()
                / (n_embd // n_head)
            )
        )
        t = torch.arange(n_seq, device=device).type_as(inv_freq)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)  # (nseq X n_embd//n_head)
        self.cos_cached = emb.cos()
        self.sin_cached = emb.sin()
        self.n_head = n_head
        self.n_seq = n_seq
        self.n_embd = n_embd
        if norm_embed:
            self.tok_emb = nn.Sequential(
                nn.Embedding(n_tok, n_embd, device=device, dtype=dtype),
                nn.LayerNorm(n_embd),
            )
        else:
            self.tok_emb = nn.Embedding(n_tok, n_embd, device=device, dtype=dtype)

    def forward(self, idx):
        return self.tok_emb(idx)

    def rotate(self, x):
        """
        Rotate along the embedding dimension.
        """
        return torch.cat([-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]], -1)

    def rotary_embed(self, q, k):
        """
        Args:
            q: A query (batch, n_head, seq, n_embd//n_head)
            k: A key. (batch, n_head, seq, n_embd//n_head)
        Returns:
            q,k (with the multiplicative rotary embedding applied.)
        """
        seq_len = q.shape[2]
        cos = self.cos_cached[None, None, :seq_len, :].to(q.device)
        sin = self.sin_cached[None, None, :seq_len, :].to(q.device)
        return (q * cos) + (self.rotate(q) * sin), (k * cos) + (self.rotate(k) * sin)


class RotarySelfAttention(nn.Module):
    """
    A self attention block with rotary relative position encoding.
    (and causality)
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.biases)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.biases)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.n_seq, config.n_seq)).view(
                1, 1, config.n_seq, config.n_seq
            ),
        )
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x, rotary_embedding: RotaryEmbedding):
        (
            B,
            T,
            C,
        ) = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q, k = rotary_embedding.rotary_embed(q, k)
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y


class RotaryBlock(nn.Module):
    """A causal, rotary Self-Attention Block."""

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = RotarySelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlpf = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.biases),
            NewGELU(),
            nn.Linear(4 * config.n_embd, config.n_embd, bias=config.biases),
        )

    def forward(self, x, rotary_embedding: RotaryEmbedding):
        x = x + self.attn(self.ln_1(x), rotary_embedding)
        x = x + self.mlpf(self.ln_2(x))
        return x


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.n_seq, config.n_seq)).view(
                1, 1, config.n_seq, config.n_seq
            ),
        )
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        (
            B,
            T,
            C,
        ) = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y


class Block(nn.Module):
    """A causal Self-Attention Block."""

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlpf = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            NewGELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
        )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x


class NonCausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        # self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
        #                              .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        (
            B,
            T,
            C,
        ) = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf')) # This is why it's non-causal.
        att = F.softmax(att, dim=-1)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.c_proj(y)
        return y


class NonCausalBlock(nn.Module):
    """A _n-causal_ Self-Attention Block."""

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = NonCausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlpf = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            NewGELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
        )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x
