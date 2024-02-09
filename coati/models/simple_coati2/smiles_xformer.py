from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import autocast
from torch.nn import functional as F

from coati.models.simple_coati2.basic_transformer import (
    RotaryEmbedding,
    RotaryBlock,
)

# -----------------------------------------------------------------------------


@dataclass
class SmilesTransformerConfig:
    n_layer: int = 4
    n_embd: int = 128
    n_head: int = 4
    n_seq: int = 256
    n_tok: int = 100
    biases: bool = True  # Whether to use biases in the linear layers.
    norm_embed: bool = False  # Whether to normalize post-embed.
    device: None = torch.device("cpu")
    dtype: None = torch.float


class SimpleTokenEmbedding(nn.Module):
    def __init__(
        self,
        n_embd=128,
        n_tok=100,
        n_seq=256,
        device=torch.device("cpu"),
        dtype=torch.float,
    ):
        super().__init__()
        self.n_embd = n_embd
        self.n_seq = n_seq
        self.pos_emb = nn.Embedding(n_seq, n_embd, device=device, dtype=dtype)
        self.tok_emb = nn.Embedding(n_tok, n_embd, device=device, dtype=dtype)

    def forward(self, x):
        return self.tok_emb(x) + self.pos_emb(
            torch.arange(0, x.shape[1], dtype=torch.uint16, device=x.device)
        ).unsqueeze(0)


def get_stop_token_embs(x, idx, tokenizer):
    """
    Args:
        x: batch X seq X hidden floattensor of logits.
        idx: batch X seq token long-tensor
        tokenizer: a tokenizer.
    """
    Is, Js = (idx == tokenizer.stop_token).nonzero(as_tuple=True)
    stop_embs = x[Is, Js]

    if not stop_embs.shape[0] == x.shape[0]:
        raise RuntimeError(
            "Some smiles in the batch do not have stop tokens. Did some tokenizations fail?"
        )

    return stop_embs
    # # this should correspond to stop token.
    # last_nonzero = (idx > 0).sum(dim=1) - 1
    # stop_token_embds = x[torch.arange(idx.shape[0]), last_nonzero, :]
    # assert stop_token_embds.shape[0] == idx.shape[0]
    # assert stop_token_embds.shape[1] == x.shape[-1]
    # return stop_token_embds


# TODO: merge these classes? Eh, the non-rot will just die off.
# Classes considered harmful.com
# Better TODO: Do the version of relative position emb from TransformerXL


class RotarySmilesTransformer(nn.Module):
    """
    Rotary string transformer for a tokenized graph
    """

    def __init__(self, config: SmilesTransformerConfig):
        super().__init__()
        self.n_seq = config.n_seq
        self.n_tok = config.n_tok
        self.n_embd = config.n_embd
        if config.norm_embed:
            self.norm_embed = nn.LayerNorm(config.n_embd)
        else:
            self.norm_embed = nn.Identity()
        self.emb = RotaryEmbedding(
            n_embd=config.n_embd,
            n_seq=config.n_seq,
            n_tok=config.n_tok,
            n_head=config.n_head,
            device=config.device,
            norm_embed=config.norm_embed,
            dtype=config.dtype,
        )
        self.transformer = nn.ModuleDict(
            dict(
                h=nn.ModuleList([RotaryBlock(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.n_tok, bias=False)

        # report number of parameters (note we don't count the decoder parameters in lm_head)
        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("number of parameters: %.2fM" % (n_params / 1e6,))

    def encode(self, idx, tokenizer):
        """
        Only returns the vector of the [STOP] token
        which MUST be the last token before [PAD]
        """
        x = self.xformer(idx)
        return get_stop_token_embs(x, idx, tokenizer)

    def generate_greedy(self, prefix=torch.tensor([[1]]), stop_token=2, max_len=256):
        """
        Autoregressively generate. c.f. https://huggingface.co/blog/how-to-generate
        Note: the decode from forward is not autoregressive.
        this is and stops upon hitting a stop token.
        """
        generated = torch.clone(prefix)
        with torch.no_grad():
            while (generated.flatten()[-1].item() != stop_token) and generated.shape[
                1
            ] < self.n_seq:
                Y = self.forward(generated, decode=False, sampled=False)
                _, next_char = torch.topk(Y[0, generated.shape[1] - 1], k=1, dim=-1)
                generated = torch.cat([generated, next_char.unsqueeze(0)], 1)
        return generated

    def generate_topk(
        self,
        prefix=torch.tensor([[1]]),
        stop_token=2,
        inv_temp=2,
        k=10,
    ):
        """
        Args:
            inj_token: (int) if not none, will perform token injection for clip gen.
            inj_hidden: torch.float tensor if not none will be injected over inj_token
        https://arxiv.org/pdf/1805.04833.pdf
        """
        generated = torch.clone(prefix).to(self.lm_head.weight.device)
        with torch.no_grad():
            while (generated.flatten()[-1].item() != stop_token) and generated.shape[
                1
            ] < self.n_seq:
                Y = self.forward(generated)  # Y is
                logits, inds = torch.topk(Y[0, generated.shape[1] - 1], k=k, dim=-1)
                probs = F.softmax(logits * inv_temp, dim=-1)
                inds_of_inds = torch.multinomial(probs, num_samples=1).squeeze()
                generated = torch.cat(
                    [generated, (inds[inds_of_inds]).unsqueeze(0).unsqueeze(0)], 1
                )
        return generated[0].tolist()

    def generate_topk_batch(
        self, prefix=[[0]], stop_token=2, pad_token=0, inv_temp=2, k=10
    ):
        """
        Works for variable length prefixes.
        """
        batch_size = len(prefix)
        min_prefix_len = min([len(p) for p in prefix])
        # fill in a zero-ed out prefix tensor.
        # which will overwrite the new columns each iteration.
        prefix_t = torch.zeros(
            (batch_size, self.n_seq),
            device=self.lm_head.weight.device,
            dtype=torch.long,
        )
        for K, row in enumerate(prefix):
            prefix_t[K, : len(row)] = torch.tensor(
                row, device=self.lm_head.weight.device, dtype=torch.long
            )

        current_t = prefix_t.clone()
        idx = min_prefix_len - 2
        has_stopped = []

        while len(has_stopped) < batch_size and idx < self.n_seq - 1:
            current_t[prefix_t > 0] = prefix_t[prefix_t > 0]
            x = self.emb(current_t)

            for block in self.transformer.h:
                x = block(x, self.emb)
            x = self.transformer.ln_f(x)
            logits = self.lm_head(x)
            logits, inds = torch.topk(logits[:, idx], k=k, dim=1)
            probs = F.softmax(logits * inv_temp, dim=1)
            inds_of_inds = torch.multinomial(probs, num_samples=1).squeeze()
            last_tokens = inds[torch.arange(batch_size), inds_of_inds]
            last_tokens[has_stopped] = pad_token

            current_t[:, idx + 1] = last_tokens

            idx += 1
            has_stopped = (current_t == stop_token).nonzero(as_tuple=True)[0]

        return current_t.tolist()

    # TODO: some consolidation here
    def xformer_blocks(
        self, x: torch.Tensor, apply_norm: bool = True, output_logits: bool = False
    ) -> torch.Tensor:
        for block in self.transformer.h:
            x = block(x, self.emb)
        if apply_norm:
            x = self.transformer.ln_f(x)

        if output_logits:
            return self.lm_head(x)
        else:
            return x

    def generate_topk_with_inj(
        self,
        prefix=[0],
        stop_token=2,
        inv_temp=1,
        k=50,  # only the topk logits can be randomly gen'd
        inj_token=None,
        inj_payload=None,
    ):
        """
        Like the above, but works in the embedding space rather than token space, so it can do
        clip injection.

        Args:
            inj_token: (int) if not none, will perform token injection for clip gen.
            inj_payload: torch.float tensor if not none will be injected over inj_token
                         [just n_hidden]
        https://arxiv.org/pdf/1805.04833.pdf
        """
        assert (
            len(prefix) <= self.n_seq
        ), f"Cannot forward sequence of length {len(prefix)}, n_seq is only {self.n_seq}"
        prefix_x = self.emb(
            torch.tensor(prefix, device=inj_payload.device, dtype=torch.long).unsqueeze(
                0
            )
        )
        # Inject the payload
        prefix_x[0, prefix.index(inj_token)] = inj_payload
        # from now on embedded vectors will be generated by concatenation along the seq dim
        generated = []
        last_token = 0
        with torch.no_grad():
            while (last_token != stop_token) and len(generated) < self.n_seq - 1:
                if len(generated):
                    # concatenate the generated tokens onto the prefix.
                    gen_x = self.emb(
                        torch.tensor(
                            generated, device=inj_payload.device, dtype=torch.long
                        ).unsqueeze(0)
                    )
                    x = torch.cat([prefix_x, gen_x], 1)
                else:
                    x = prefix_x
                for block in self.transformer.h:
                    x = block(x, self.emb)
                x = self.transformer.ln_f(x)
                logits = self.lm_head(x)
                logits, inds = torch.topk(
                    logits[0, len(prefix) + len(generated) - 1], k=k, dim=-1
                )
                probs = F.softmax(logits * inv_temp, dim=-1)
                inds_of_inds = torch.multinomial(probs, num_samples=1).squeeze()
                last_token = inds[inds_of_inds].item()
                generated.append(last_token)
        return prefix + generated

    def generate_top_k_with_inj_batch(
        self,
        prefix=[0],
        stop_token=2,
        pad_token=0,
        inv_temp=1,
        k=50,  # only the topk logits can be randomly gen'd
        inj_token=None,
        inj_payload=None,
        as_tensor=False,
    ):
        batch_size = inj_payload.size(0)
        assert inj_payload.dim() == 2
        assert inj_payload.shape[-1] == self.n_embd
        assert k >= 1
        prefix_x = (
            self.emb(torch.tensor(prefix, device=inj_payload.device, dtype=torch.long))
            .unsqueeze(0)
            .repeat(batch_size, 1, 1)
        )
        # Inject the payload
        prefix_x[:, prefix.index(inj_token), :] = inj_payload
        generated = torch.tensor([], dtype=torch.int64, device=inj_payload.device)
        has_stopped = []
        idx = 0
        # NOTE (eddie): fixed issue where if a batch did not stop
        # before hitting (seq_len - len(prefix)) it would blow up batch
        while len(has_stopped) < batch_size and idx < self.n_seq - len(prefix):
            if idx > 0:
                gen_x = self.emb(generated)
                x = torch.cat([prefix_x, gen_x], 1)
            else:
                x = prefix_x
            # batch x len(seq) x num_tokens
            logits = self.xformer_blocks(x, apply_norm=True, output_logits=True)
            # logits->batch_size x k , inds-> batch_size x k
            logits_topk, inds_topk = torch.topk(
                logits[:, len(prefix) + idx - 1], k=k, dim=1
            )
            probs = F.softmax(logits_topk * inv_temp, dim=1)
            inds_of_inds = torch.multinomial(probs, num_samples=1).squeeze(-1)
            last_tokens = inds_topk[torch.arange(batch_size), inds_of_inds]
            # if any of the batch has stopped, set their last token to pad, don't want to generate anymore
            last_tokens[has_stopped] = pad_token
            if len(generated):
                generated = torch.cat(
                    [generated, last_tokens.clone().unsqueeze(1)], dim=1
                ).long()
            else:
                generated = last_tokens.clone().unsqueeze(1).long()
            idx += 1
            has_stopped = (generated == stop_token).nonzero(as_tuple=True)[0]

        # if anything hasn't stopped yet by the time it reaches the threshold, add stop token
        num_not_stopped = batch_size - len(has_stopped)

        if num_not_stopped:
            # print(f"WARNING: {num_not_stopped} sequences did not stop before reaching max length. forcing stop.")
            # get all indices that are not in has_stopped
            not_stopped = torch.tensor(
                [i for i in range(batch_size) if i not in has_stopped]
            )
            # create a final last_token that's all pad_token except for the not_stopped
            final_pad_or_stopped = torch.tensor(
                [pad_token] * batch_size, device=inj_payload.device
            )
            final_pad_or_stopped[not_stopped] = stop_token
            # set their last token to stop token
            generated[not_stopped, -1] = stop_token
            # generated = torch.cat([generated, final_pad_or_stopped.clone().unsqueeze(1)], dim=1).long()

        if as_tensor:
            return torch.cat(
                [
                    torch.tensor(prefix, dtype=torch.long, device=generated.device)
                    .unsqueeze(0)
                    .repeat(batch_size, 1),
                    generated,
                ],
                dim=1,
            )

        token_batch = [prefix + output for output in generated.tolist()]
        return token_batch

    def xformer(self, idx):
        """
        Args:
            idx: torch longtensor of token indices.

        Returns encoding of all entries in batch.
        """
        _, t = idx.size()
        assert (
            t <= self.n_seq
        ), f"Cannot forward sequence of length {t}, n_seq is only {self.n_seq}"
        x = self.emb(idx)
        for block in self.transformer.h:
            x = block(x, self.emb)
        x = self.transformer.ln_f(x)
        return x

    def decode_logits(self, logits):
        probs = F.softmax(logits, dim=-1)
        _, idx_next = torch.topk(probs, k=1, dim=-1)
        return logits, idx_next.squeeze()

    def forward(self, idx):
        """
        Args:
            idx: torch longtensor of token indices.
        """
        x = self.xformer(idx)
        logits = self.lm_head(x)
        return logits

    def forward_with_stop_emb(self, idx, tokenizer):
        """
        I made this a separate routine because of issues with torch DataParallel
        and functions with variable numbers of return values.
        Args:
            idx: torch longtensor of token indices.
        """
        x = self.xformer(idx)
        logits = self.lm_head(x)
        return logits, get_stop_token_embs(x, idx, tokenizer)

    def forward_with_stop_emb_and_replacement(
        self, idx, injection, tokenizer, inject_token="[UNK]"
    ):
        """
        This is specifically for e2e-CLIP a-la clipCAP.
        It injects tokens in place of
        special_token and also returns the stop-emb.

        Args:
            idx: torch longtensor of token indices. (batch X seq)
            injection: (batch X seq X emb_dim)
        """
        _, t = idx.size()
        assert (
            t <= self.n_seq
        ), f"Cannot forward sequence of length {t}, n_seq is only {self.n_seq}"
        x = self.emb(idx)
        # Injection of the special tokens
        with autocast(enabled=False, device_type="cuda"):
            hole_Is, hole_Js = (idx == tokenizer.vocab[inject_token]).nonzero(
                as_tuple=True
            )
            if torch.numel(hole_Js) > 0:
                x[hole_Is, hole_Js] = injection[hole_Is]
        # regular old xformer.
        for block in self.transformer.h:
            x = block(x, self.emb)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits, get_stop_token_embs(x, idx, tokenizer)

    def forward_with_replacement(self, idx, injection, tokenizer, inject_token="[UNK]"):
        """
        This is specifically for e2e-CLIP a-la clipCAP.
        It injects tokens in place of special_token
        SORRY FOR THE REPETITION.
        YOU KNOW ABOUT DUMB VARIABLE RETURN NUMBER ISSUES
        IN TORCH JIT  JAP-3-29-2023

        Args:
            idx: torch longtensor of token indices. (batch X seq)
            injection: (batch X seq X emb_dim)
        """
        _, t = idx.size()
        assert (
            t <= self.n_seq
        ), f"Cannot forward sequence of length {t}, n_seq is only {self.n_seq}"
        x = self.emb(idx)
        # Injection of the special tokens
        with autocast(enabled=False, device_type="cuda"):
            hole_Is, hole_Js = (idx == tokenizer.vocab[inject_token]).nonzero(
                as_tuple=True
            )
            x[hole_Is, hole_Js] = injection[hole_Is]
        # regular old xformer.
        for block in self.transformer.h:
            x = block(x, self.emb)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits


# def smiles_xform(batch, tokenizer,
#                         device = torch.device('cpu'),
#                         ):
#     """
#     Will be deprecated, for fill_in_middle or clip's xform

#     Note: this exploits the fact that the encoding will always begin with [GRAPH]
#     and end with [STOP] [PAD] [PAD] ... and that [PAD] is always token mapping to 0
#     """
#     batch['tokens'] = []
#     for S in batch['smiles']:
#         try:
#             ttext = tokenizer.tokenize_text('[SMILES]'+S+'[STOP]')
#             if (len(ttext) == tokenizer.n_seq):
#                 batch['tokens'].append(ttext)
#             else:
#                 continue
#         except Exception as Ex:
#             # print('Tokenize failure:', S, ttext, len(ttext), Ex)
#             batch['tokens'].append(torch.zeros(tokenizer.n_seq, dtype = torch.uint16))

#     for col in ['tokens']:
#         if col in batch:
#             if type(batch[col]) != torch.Tensor:
#                 batch[col] = torch.tensor(batch[col], requires_grad=False).to(device, torch.uint16)
#     # decrease the sequence size to max demanded by this batch.
#     batch['tokens'] = batch['tokens'][:,:(batch['tokens'].sum(0)>0).sum()]

#     # Alignment in cross entropy:
#     # [GRAPH],  [TOKEN 1]... [STOP] [PAD]
#     # [TOKEN 1] [TOKEN 2]... [PAD]  [PAD]
#     batch['y_next'] = torch.zeros_like(batch['tokens'])
#     batch['y_next'][:,:(batch['tokens'].shape[1]-1)] = batch['tokens'][:,1:].clone()
#     batch['y_next'][batch['y_next']==0] = -1
#     return batch
