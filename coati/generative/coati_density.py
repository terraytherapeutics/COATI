# Estimate the density of the vectors in chembl.
from typing import Iterable, Union

import torch
from rdkit import Chem
from torch.distributions.multivariate_normal import MultivariateNormal

from coati.common.util import batch_indexable
from coati.models.encoding.clip_e2e import e3gnn_smiles_clip_e2e
from coati.models.encoding.tokenizers.trie_tokenizer import TrieTokenizer


def estimate_density_batchwise(
    iterable: Iterable[str],
    encoder: e3gnn_smiles_clip_e2e,
    tokenizer: TrieTokenizer,
    batch_size: int = 1024,
    epochs: int = 10,
    entropy_limit: float = -100,  # convergence criteria
) -> Union[MultivariateNormal, None]:
    """
    Simple batchwise density estimation of multivariate normal
    distribution of embedding space given SMILES strings.
    """
    mean_param = torch.nn.Parameter(
        torch.zeros(encoder.embed_dim, device=encoder.device)
    )
    sqrt_diag_param = torch.nn.Parameter(
        0.5 * torch.ones(encoder.embed_dim, device=encoder.device)
    )  # has to be positive.
    lower_diag_indices = torch.tril_indices(
        encoder.embed_dim, encoder.embed_dim, offset=-1
    )
    lower_tri_param = torch.nn.Parameter(
        torch.zeros(lower_diag_indices.shape[-1], device=encoder.device)
    )

    def build_distribution(
        sq_diag: torch.Tensor, lower_tri: torch.Tensor
    ) -> MultivariateNormal:
        scale_tril = torch.diag(sq_diag * sq_diag)
        scale_tril[lower_diag_indices[0], lower_diag_indices[1]] = lower_tri
        density = MultivariateNormal(mean_param, scale_tril=scale_tril)
        return density

    optimizer = torch.optim.SGD([sqrt_diag_param, lower_tri_param], lr=5e-3)

    for k in range(epochs):
        batches = batch_indexable(iterable, batch_size)
        for batch in batches:
            canonical_smiles = []
            batch_tokens_ = []
            for S in batch:
                try:
                    canonical_smiles.append(Chem.MolToSmiles(Chem.MolFromSmiles(S)))
                    batch_tokens_.append(
                        tokenizer.tokenize_text(
                            "[SMILES]" + canonical_smiles[-1] + "[STOP]", pad=True
                        )
                    )
                except:
                    continue
            with torch.no_grad():
                batch_tokens = torch.tensor(
                    batch_tokens_, device=encoder.device, dtype=torch.int
                )
                batch_embeds = encoder.encode_tokens(batch_tokens, tokenizer)
            distribution = build_distribution(sqrt_diag_param, lower_tri_param)
            entropy = -1 * distribution.log_prob(batch_embeds).mean()
            print(f"entropy: {entropy.detach().cpu().item():.4f}")
            if entropy.detach().cpu().item() < entropy_limit:
                return distribution
            optimizer.zero_grad()
            entropy.backward()
            optimizer.step()
    return None
