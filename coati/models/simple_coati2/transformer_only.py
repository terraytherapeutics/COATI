import numpy as np
from rdkit import Chem
from rdkit import RDLogger
import torch
import torch.nn as nn
from torch.nn import functional as F

from coati.models.simple_coati2.smiles_xformer import (
    SmilesTransformerConfig,
    RotarySmilesTransformer,
)


RDLogger.DisableLog("rdApp.*")
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


class SwiGLUResNet(nn.Module):
    def __init__(self, d_in, d_out, dropout=0.0):
        """
        10/25 - added dropout.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_in),
            torch.nn.Dropout(p=dropout),
            nn.Linear(d_in, 2 * d_out),
            SwiGLU(),
            nn.Linear(d_out, d_out),
        )

    def forward(self, x):
        return self.net(x) + x


class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return torch.nn.functional.silu(gate) * x


class COATI_Smiles_Inference(nn.Module):
    """
    A coati that can try to take advantage of the
    pseudoscalar signal from allegro.
    """

    def __init__(
        self,
        n_layer_xformer=16,
        n_hidden_xformer=256,
        embed_dim=256,
        n_head=16,
        n_seq=80,
        mlp_dropout=0.0,
        enc_to_coati="linear",  # 'swiglu_mlp',
        n_direct_clr=64,  # n_dim to take from the representation for the directCLR loss.
        n_tok=4,  # I think this is a hack to pickle num toks processed
        biases=True,
        device=torch.device("cpu"),
        dtype=torch.float,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.enc_to_coati = enc_to_coati
        self.n_direct_clr = n_direct_clr

        kwargs = {
            "n_layer": n_layer_xformer,
            "n_embd": n_hidden_xformer,
            "n_head": n_head,
            "n_seq": n_seq,
            "n_tok": n_tok,
            "device": device,
            "dtype": dtype,
            "biases": biases,
        }

        self.xformer_config = SmilesTransformerConfig(**kwargs)
        self.xformer = RotarySmilesTransformer(self.xformer_config)
        self.device = device

        if enc_to_coati == "linear":
            self.smiles_to_coati = nn.Sequential(
                nn.LayerNorm(self.embed_dim),
                nn.Linear(self.xformer.n_embd, self.embed_dim),
            )
        # Make the common representation
        elif enc_to_coati == "swiglu_mlp":
            self.smiles_to_coati = nn.Sequential(
                nn.LayerNorm(self.xformer.n_embd),
                nn.Linear(self.xformer.n_embd, 2 * self.embed_dim),
                SwiGLU(),
                nn.Linear(self.embed_dim, self.embed_dim),
            )
        elif enc_to_coati == "swiglu_resnet":
            self.smiles_to_coati = SwiGLUResNet(
                self.xformer.n_embd, self.embed_dim, dropout=mlp_dropout
            )

        self.coati_to_token = SwiGLUResNet(self.embed_dim, self.embed_dim)

        n_params_smiles = sum(p.numel() for p in self.xformer.parameters())
        print(f"number of parameters Total: xformer: {n_params_smiles/1e6:.2f}M ")
        self.to(self.device)

    def encode_tokens(self, token_indices, tokenizer):
        assert token_indices.dim() == 2
        return self.smiles_to_coati(self.xformer.encode(token_indices, tokenizer))

    def hcoati_to_2d(
        self,
        h_coati,
        tokenizer,
        fill_in_from="[SMILES]",
        noise_scale=0.0,
        do_suffix=False,
        inv_temp=2,
        k=100,
    ):
        """
        Testing generation of SMILES (or GRAPH)
        from atoms and coords
        """
        assert fill_in_from == "[SMILES]" or fill_in_from == "[GRAPH]"
        if noise_scale > 0:
            h_coati += torch.normal(
                mean=torch.zeros_like(h_coati),
                std=noise_scale * torch.ones_like(h_coati),
            )
        h_token = self.coati_to_token(h_coati)
        # create a 'batch' to infer smiles.
        if do_suffix:
            suffstr = "[SUFFIX][MIDDLE]"
        else:
            suffstr = ""
        token_prebatch = tokenizer.tokenize_text(
            "[CLIP][UNK]" + fill_in_from + suffstr, pad=False
        )
        generation = self.xformer.generate_topk_with_inj(
            prefix=token_prebatch,
            stop_token=tokenizer.stop_token,
            inv_temp=inv_temp,
            k=k,
            inj_token=tokenizer.unk_token,
            inj_payload=h_token[0],
        )
        if fill_in_from == "[SMILES]":
            return tokenizer.decode(generation, special=False)
        else:
            return tokenizer.decode(generation)

    def hcoati_to_2d_batch(
        self,
        h_coati: torch.Tensor,
        tokenizer,
        fill_in_from: str = "[SMILES]",
        noise_scale: float = 0.0,
        inv_temp: float = 2,
        k: int = 100,
        do_suffix=False,
        keep_special: bool = False,
        return_tokens: bool = False,
    ):
        """
        Testing generation of SMILES (or GRAPH)
        from atoms and coords
        """
        assert k > 1
        if noise_scale > 0:
            h_coati += torch.normal(
                mean=torch.zeros_like(h_coati),
                std=noise_scale * torch.ones_like(h_coati),
            )
        h_token = self.coati_to_token(h_coati)
        if do_suffix:
            suffstr = "[SUFFIX][MIDDLE]"
        else:
            suffstr = ""
        token_prebatch = tokenizer.tokenize_text(
            "[CLIP][UNK]" + fill_in_from + suffstr, pad=False
        )
        assert h_token.dim() == 2
        assert h_token.shape[-1] == self.xformer.n_embd
        generation = self.xformer.generate_top_k_with_inj_batch(
            prefix=token_prebatch,
            stop_token=tokenizer.stop_token,
            inv_temp=inv_temp,
            k=k,
            pad_token=tokenizer.pad_token,
            inj_token=tokenizer.unk_token,
            inj_payload=h_token,
        )
        smiles_list = [
            tokenizer.decode(token_out, special=keep_special)
            for token_out in generation
        ]

        if return_tokens:
            return smiles_list, generation

        return smiles_list
