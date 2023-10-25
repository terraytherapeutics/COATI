import numpy as np
import torch
from typing import List
from rdkit import Chem

from coati.containers.rdkit_utils import mol_to_atoms_coords
from coati.models.encoding.clip_e2e import e3gnn_smiles_clip_e2e
from coati.models.encoding.tokenizers.trie_tokenizer import TrieTokenizer


def embed_points(s: str, encoder: e3gnn_smiles_clip_e2e) -> torch.Tensor:
    V0_atoms, V0_coords = mol_to_atoms_coords(s)
    with torch.no_grad():
        V0 = torch.from_numpy(
            encoder.encode_points(
                torch.tensor(V0_atoms, device=encoder.device).unsqueeze(0).float(),
                torch.tensor(V0_coords, device=encoder.device).unsqueeze(0).float(),
            )
            .detach()
            .cpu()
            .numpy()
        ).cuda()
    return V0


def embed_smiles(
    s: str, encoder: e3gnn_smiles_clip_e2e, tokenizer: TrieTokenizer
) -> torch.Tensor:
    s = Chem.MolToSmiles(Chem.MolFromSmiles(s))
    with torch.no_grad():
        try:
            batch_tokens = torch.tensor(
                [tokenizer.tokenize_text("[SMILES]" + s + "[STOP]", pad=True)],
                device=encoder.device,
                dtype=torch.int,
            )
            batch_embeds = encoder.encode_tokens(batch_tokens, tokenizer)
        except Exception as Ex:
            print("embed_smiles exception: ", Ex)
    return batch_embeds[0]

def embed_smiles_batch(smiles_list: List[str], encoder: e3gnn_smiles_clip_e2e, tokenizer: TrieTokenizer) -> torch.Tensor:
    batch_tokens = torch.tensor(
        [tokenizer.tokenize_text("[SMILES]" + s + "[STOP]", pad=True) for s in smiles_list],
        device=encoder.device,
        dtype=torch.int,
    )
    batch_embeds = encoder.encode_tokens(batch_tokens, tokenizer)
    return batch_embeds

def purify_vector(
    V: torch.Tensor, encoder: e3gnn_smiles_clip_e2e, tokenizer: TrieTokenizer, n_rep=128
) -> torch.Tensor:
    """
    purification is usually the name given to an operation which
    pulls out the idempotent part of a vector under a map.

    for example gs dm satisfies P**2 - P =0
    Purification of density matrix:
        min(tr((P^2-p)**2))

    Can we purify a coati vector? The issue is the decoding process
    isn't deterministic or differentiable.
    We would like to ensure:
        vector  = embed(decode(vector)) which is also:
        0 = embed(decode(vector)) - vector

    Can we enforce this via a gradient like step?
    I'm going to try the punt-version which just pushes vector
    towards the average of embed(decode(vector))

    Args:
        V (batch X embed_dim)
    """
    with torch.no_grad():
        try:
            regen_smiles = encoder.hclip_to_2d_batch(
                V.to(encoder.device).unsqueeze(0).repeat(n_rep, 1), tokenizer
            )
        except Exception as Ex:
            return V
        batch_tokens_ = []
        for S in regen_smiles:
            try:
                S = Chem.MolToSmiles(Chem.MolFromSmiles(S))
                batch_tokens_.append(
                    tokenizer.tokenize_text("[SMILES]" + S + "[STOP]", pad=True)
                )
            except Exception as Ex:
                pass
        if len(batch_tokens_) < 1:
            return V
        batch_tokens = torch.tensor(
            batch_tokens_, device=encoder.device, dtype=torch.long
        )
        batch_embeds = encoder.encode_tokens(batch_tokens, tokenizer)
        return batch_embeds.mean(0)


def force_decode_valid(
    V: torch.Tensor,
    encoder: e3gnn_smiles_clip_e2e,
    tokenizer: TrieTokenizer,
    max_attempts: int = 2000,
) -> str:
    """
    Continues decoding until a valid SMILES string is produced.
    """
    for attempt in range(max_attempts):
        with torch.no_grad():
            try:
                regen_smiles = encoder.hclip_to_2d(V, tokenizer)
                mol = Chem.MolFromSmiles(regen_smiles)
                if not mol is None:
                    return regen_smiles
            except Exception as Ex:
                #                 print(Ex)
                pass
    return "C"


def force_decode_valid_batch(
    V: torch.Tensor,
    encoder: e3gnn_smiles_clip_e2e,
    tokenizer: TrieTokenizer,
    batch_size: int = 128,
    max_attempts: int = 4,
) -> str:
    """
    Attemps multiple parallel decodings until a valid SMILES string is produced.
    If multiple valid SMILES strings are produced, returns the most common one.
    """
    for k in range(max_attempts):
        try:
            with torch.no_grad():
                regen_smiles = encoder.hclip_to_2d_batch(
                    V.unsqueeze(0).repeat(batch_size, 1), tokenizer
                )
                slist = []
                for S in regen_smiles:
                    try:
                        mol = Chem.MolFromSmiles(S)
                        if not mol is None:
                            slist.append(Chem.MolToSmiles(mol))
                    except Exception as Ex:
                        print(Ex)
                        pass
            if len(slist):
                return slist[np.argmax([slist.count(S) for S in slist])]
            else:
                continue
        except Exception as Ex:
            continue
    return "C"
