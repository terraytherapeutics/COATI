"""
This version only passes nonzero messages. 
"""
from typing import Callable, Tuple

import torch
from torch import nn


def cubic_cutoff(
    x: torch.Tensor, y=torch.tensor(5.0, dtype=torch.float, requires_grad=False)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    f(y) = 0, f'(y)=0, f(0)=1, f'(0)=0
    f(r) = 1 + (-3/2)r_c^{-2}r^2 + (1/2)r_c^{-3}r^3
    """
    assert y > 0
    a = 1.0
    c = (-3.0 / 2) * torch.pow(y, -2).to(dtype=x.dtype)
    d = (1.0 / 2) * torch.pow(y, -3).to(dtype=x.dtype)
    x_cut = a + c * torch.pow(x, 2.0) + d * torch.pow(x, 3.0)
    return torch.where(
        x <= 0, torch.ones_like(x), torch.where(x >= y, torch.zeros_like(x), x_cut)
    )


def make_neighborlist(
    x: torch.Tensor,
    node_mask: torch.Tensor,
    cutoff=torch.tensor(5.0, requires_grad=False),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Construct a list of neighbors for each node in each graph
    no grad! I = batch index, J = atom 1, K = atom 2.

    Args:
        x: n_graph X n_node X 3
        node_mask: n_graph X n_node
    Returns:
        (n_neighbors) indices (Is, Js, Ks)
        such that x[Is,Js] => partner one coords.
                  x[Is,Ks] => partner two coords.
    """
    n_batch = x.shape[0]
    n_node = x.shape[1]
    d = torch.cdist(x, x)
    pair_mask = node_mask.unsqueeze(1).tile(1, n_node, 1) * node_mask.unsqueeze(2).tile(
        1, 1, n_node
    )
    in_range = torch.logical_and(
        (d < cutoff.to(x.device)),
        torch.logical_not(
            torch.eye(x.shape[1], dtype=torch.bool, device=x.device)
            .unsqueeze(0)
            .repeat(n_batch, 1, 1)
        ),
    )
    whole_mask = torch.logical_and(pair_mask, in_range)
    Is = (
        torch.arange(n_batch, device=x.device, dtype=torch.long)
        .unsqueeze(-1)
        .unsqueeze(-1)
        .repeat(1, n_node, n_node)[whole_mask]
    )
    Js = (
        torch.arange(x.shape[1], device=x.device, dtype=torch.long)
        .unsqueeze(0)
        .unsqueeze(-1)
        .repeat(n_batch, 1, n_node)[whole_mask]
    )
    Ks = (
        torch.arange(x.shape[1], device=x.device, dtype=torch.long)
        .unsqueeze(0)
        .unsqueeze(0)
        .repeat(n_batch, n_node, 1)[whole_mask]
    )
    return Is, Js, Ks, d[Is, Js, Ks]


class e_gcl_sparse(nn.Module):
    """
    Equivariant Graph Convolutional layer
    this differs from the welling implementation in
     - avoids expensive & wasteful indexing between batch pairs.
     - uses sparsity of cutoff in neighborlists.
     - avoids edge models.
    (3)-(6) of 2102.09844 equations shown below

    note: the messages are SMOOTHLY cutoff.
    """

    def __init__(
        self,
        input_nf: int,
        output_nf: int = None,
        hidden_nf: int = None,
        act_fn: Callable = nn.SiLU(),
        recurrent: bool = True,
        residual: bool = True,
        attention: bool = False,
        instance_norm: bool = False,
        residual_nf: int = 0,
        message_cutoff: float = 5.0,
        dropout: float = 0.0,
        prop_coords: bool = True,
    ):
        super(e_gcl_sparse, self).__init__()
        self.message_cutoff = torch.tensor(message_cutoff, requires_grad=False)
        if output_nf is None:
            output_nf = input_nf
        if hidden_nf is None:
            hidden_nf = input_nf
        self.residual_nf = residual_nf

        input_edge = input_nf * 2

        self.residual = residual
        self.recurrent = recurrent
        self.attention = attention
        self.prop_coords = prop_coords
        if instance_norm:
            self.instance_norm = torch.nn.InstanceNorm1d(hidden_nf)
        else:
            self.instance_norm = torch.nn.Identity()
        self.dropout = dropout

        edge_coords_nf = 1  # Squared distance in (3)

        # \phi_e of eqn(3)
        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf, hidden_nf),
            act_fn,
            nn.Dropout(p=self.dropout) if self.dropout else nn.Identity(),
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            nn.Dropout(p=self.dropout) if self.dropout else nn.Identity(),
        )

        # \phi_h of eqn(6)
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + self.residual_nf + input_nf, hidden_nf),
            act_fn,
            nn.Dropout(p=self.dropout) if self.dropout else nn.Identity(),
            nn.Linear(hidden_nf, output_nf),
        )

        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        self.coord_mlp = nn.Sequential(*coord_mlp)

        # This is a learned edge mask basically.
        # I recommend it's not used.
        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(input_nf, hidden_nf),
                act_fn,
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid(),
            )

        self.act_fn = act_fn
        return

    def edge_model(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        node_mask: torch.Tensor,
        distance_gradient: bool = False,
        debug: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Eq(3) (now done dense, no node attributes)

        Args:
            h: batch X natom X nhidden
            x: batch X natom X 3
            node_mask: batch X natom (True = Nonzero)
        Returns:
            mij: batch X nmsg X nhidden
            Is: nmsg
            Js: nmsg
            Ks: nmsg
            Ds: nmsg
        """
        nb = h.shape[0]
        na = h.shape[1]
        na2 = na * na
        nh = h.shape[2]
        nc = x.shape[-1]
        assert x.shape[0] == h.shape[0]  # same number of atoms.
        assert x.shape[1] == h.shape[1]  # same number of atoms.

        if distance_gradient:
            Is, Js, Ks, Ds = make_neighborlist(x, node_mask, self.message_cutoff)
        else:
            with torch.no_grad():
                Is, Js, Ks, Ds = make_neighborlist(x, node_mask, self.message_cutoff)
        h2 = torch.cat([h[Is, Js, :], h[Is, Ks, :], (Ds * Ds).unsqueeze(-1)], -1)
        msg_mask = cubic_cutoff(Ds, self.message_cutoff).unsqueeze(-1)
        # Mask outputs before and after attention.
        mij = self.edge_mlp(h2) * msg_mask

        if self.attention:
            att_val = self.att_mlp(mij)
            mij = mij * att_val * msg_mask
        if debug:
            print(att_val.sum())
            print("msg grad", torch.autograd.grad(mij.sum(), x, retain_graph=True))
        return mij, Is, Js, Ks, Ds

    def coord_model(
        self,
        x: torch.Tensor,
        mij: torch.Tensor,
        Is: torch.Tensor,
        Js: torch.Tensor,
        Ks: torch.Tensor,
        Ds: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Eqs (4)

        Args:
            x: batch X natom X 3
            mij: nmsg X nhidden
            Is, Js, Ks, Ds: nmsg (indices and distances for non-masked pairs)
            node_mask: batch X natom (True = Nonzero)
        Returns:
            new x: batch X natom X 3
        """
        nb = x.shape[0]
        na = x.shape[1]
        na2 = na * na
        nh = mij.shape[-1]
        nc = x.shape[-1]
        assert na > 1
        C = 1.0 / (na - 1.0)

        phi_x_mij = self.coord_mlp(mij)
        x_update = torch.zeros(nb, na, 3, dtype=x.dtype, device=x.device)
        x_update[Is, Js, :] += C * (x[Is, Js, :] - x[Is, Ks, :]) * phi_x_mij
        out = x + x_update

        return torch.clamp(out, -1000.0, 1000.0)

    def node_model(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        mij: torch.Tensor,
        Is: torch.Tensor,
        Js: torch.Tensor,
        Ks: torch.Tensor,
        Ds: torch.Tensor,
        node_mask: torch.Tensor,
        h0: torch.Tensor,
    ) -> torch.Tensor:
        """
        Eqs (5),(6)

        Args:
            h: batch X natom X nhidden
            x: batch X natom X 3
            mij: nmsg X nhidden
            Is, Js, Ks, Ds: nmsg (indices and distances.)
            node_mask: batch X natom (True = Nonzero)
            h0: if residual_nf > 0: this is the residual-like node h0
        """
        nb = h.shape[0]
        na = h.shape[1]
        na2 = na * na
        nh = h.shape[2]
        nc = x.shape[-1]
        assert x.shape[0] == h.shape[0]  # same number of atoms.
        if self.residual_nf > 0:
            assert not h0 is None
        mi = (
            torch.zeros(nb * na, nh, device=h.device, dtype=h.dtype)
            .scatter_add_(0, (na * Is + Js).unsqueeze(-1).tile(1, nh), mij)
            .reshape(nb, na, nh)
        )
        if self.residual_nf:
            out = self.node_mlp(torch.cat([h, mi, h0], dim=-1))
        else:
            out = self.node_mlp(torch.cat([h, mi], dim=-1))
        if self.recurrent:
            out = h + out
        return out

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        node_mask: torch.Tensor,
        h0: torch.Tensor,
        distance_gradient: bool = False,
        debug: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h: n_graph X n_node X n_hidden
            x: n_graph X n_node X 3
            node_mask: (unused in welling.)
        """
        mij, Is, Js, Ks, Ds = self.edge_model(
            h, x, node_mask, distance_gradient=distance_gradient, debug=debug
        )
        h_new = self.instance_norm(
            self.node_model(h, x, mij, Is, Js, Ks, Ds, node_mask, h0)
        )
        if debug:
            print("hgrad", torch.autograd.grad(h_new.sum(), x, retain_graph=True))
        x_new = self.coord_model(x, mij, Is, Js, Ks, Ds, node_mask)
        return h_new, x_new
