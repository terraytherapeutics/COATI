"""
A simplified version of e3gnn 
because CLIP is complex enough :O
No decoder, no coordinates
Just dumps the hidden rep. 
"""
import torch
import torch.nn as nn

from coati.common.periodic_table import XY_ONE_HOT_FULL
from coati.models.encoding.e_gcl_sparse import e_gcl_sparse


class e3gnn_clip(torch.nn.Module):
    def __init__(
        self,
        in_node_nf: int = len(XY_ONE_HOT_FULL(1)),
        hidden_nf: int = 128,
        device: str = "cpu",
        act_fn: str = "SiLU",
        n_layers: int = 5,
        instance_norm: bool = True,
        message_cutoff: int = 5,
        dtype=torch.float,
        torch_emb: bool = False,
        residual: bool = False,
        dropout: float = 0.1,
    ):
        """
        The Welling research code is quadratic in batch size.
        and has no instancenorm. This fixes that.
        This also has no edge feature b/c bonds aren't real

        h_l => n_graph X n_node X n_hidden_features
        x_l => n_graph X n_node X n_dim
        e_ij => n_graph X n_node X n_node X n_edge_features

        Args:
            in_node_nf: number of input features for each node (atom)
            in_edge_nf: number of input featuers for each edge (bond)
            hidden_nf: dimension of the hidden representation (per atom)
            code_nf: dimension of a code conditioning the final aggregation. (optional)
            residual_feature: whether to include residual-like h0 in the node_model
        """
        super(e3gnn_clip, self).__init__()
        self.dtype = dtype
        self.hidden_nf = hidden_nf

        if not torch_emb:
            self.torch_emb = False
            self.in_node_nf = in_node_nf
            self.emb = None
        else:
            self.torch_emb = True
            self.in_node_nf = hidden_nf
            self.emb = nn.Embedding(84, self.hidden_nf, device=device, dtype=dtype)

        self.device = device
        self.n_layers = n_layers
        self.instance_norm = instance_norm
        self.message_cutoff = torch.tensor(message_cutoff, requires_grad=False)

        assert dropout >= 0.0 and dropout < 1.0
        self.dropout = dropout

        if act_fn == "SiLU":
            self.act_fn = nn.SiLU()
        elif act_fn == "GELU":
            self.act_fn = nn.GELU()
        else:
            raise Exception("Bad act_fn")

        ### Encoder
        if self.torch_emb:
            self.embedding = torch.nn.Identity()
        else:
            self.embedding = nn.Linear(self.in_node_nf, hidden_nf)

        if instance_norm:
            self.embedding_norm = torch.nn.InstanceNorm1d(hidden_nf)
        else:
            self.embedding_norm = torch.nn.Identity()

        self.node_dec = nn.Sequential(
            nn.Linear(self.hidden_nf, self.hidden_nf),
            self.act_fn,
            nn.Dropout(p=self.dropout) if self.dropout else nn.Identity(),
            nn.Linear(self.hidden_nf, self.hidden_nf),
        )

        for i in range(0, n_layers):
            self.add_module(
                "gcl_%d" % i,
                e_gcl_sparse(
                    self.hidden_nf,
                    act_fn=self.act_fn,
                    residual=residual,
                    attention=False,
                    instance_norm=instance_norm,
                    residual_nf=(in_node_nf if residual else 0),
                    dropout=dropout,
                    prop_coords=False,
                ),
            )

        self.to(self.device)

    def forward(self, atoms: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """
        atoms: batch X max_n_atom long tensor of atomic numbers.
        coords: node coordinates.
        """
        if self.torch_emb:
            assert (atoms > 84).sum().detach().item() == 0
            nodes = self.emb(atoms)
        else:
            with torch.no_grad():
                ans = atoms.tolist()
                nodes = torch.tensor(
                    [[XY_ONE_HOT_FULL(int(atom)) for atom in mol] for mol in ans],
                    dtype=torch.float32,
                    device=atoms.device,
                    requires_grad=False,
                )
        node_mask = (atoms > 0).to(atoms.device, torch.float)
        assert nodes.isfinite().all()
        assert coords.isfinite().all()
        assert node_mask.isfinite().all()
        # bsize x n_atoms x hidden_nf
        h = self.embedding_norm(self.embedding(nodes))
        for i in range(0, self.n_layers):
            h, _ = self._modules["gcl_%d" % i](h, coords, node_mask, h0=nodes)
        h = self.node_dec(h)
        h = h * node_mask.unsqueeze(-1)
        natoms = torch.maximum(node_mask.sum(-1), torch.ones_like(node_mask.sum(-1)))
        h = torch.sum(h, dim=1) / natoms.unsqueeze(-1)
        return h
