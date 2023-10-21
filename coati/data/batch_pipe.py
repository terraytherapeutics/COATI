from torch.utils.data.datapipes.datapipe import IterDataPipe
from torch.utils.data.datapipes._decorator import functional_datapipe
import hashlib
import pandas as pd
import numpy as np
import pickle


def stack_batch(rows, return_coords=True, return_grads=False, return_dipole=False):
    """
    Stack arrays when needed.
    """
    batch = {}
    if return_coords:
        # batch and pad atom arrays
        nrows = len(rows)
        natoms = [X["atoms"].shape[0] if "atoms" in X else 0 for X in rows]
        max_atoms = np.max(natoms)
        atoms = np.zeros((nrows, max_atoms))
        coords = np.zeros((nrows, max_atoms, 3))
        if return_grads:
            return_grads = True
            grads = np.zeros((nrows, max_atoms, 3))
        if return_dipole:
            return_dipole = True
            dipoles = np.zeros((nrows, 3))
        else:
            return_dipole = False

        for i, row in enumerate(rows):
            if "atoms" in row:
                row_atoms = row["atoms"]
                row_coords = row["coords"]

                if return_grads and "gradients" in row:
                    row_grads = row["gradients"]
                    grads[i, : row_grads.shape[0], :] = row_grads.copy()

                if return_dipole and "dipole" in row:
                    row_dipole = row["dipole"]
                    dipoles[i, :] = row_dipole.copy()

                try:
                    atoms[i, : row_atoms.shape[0]] = row_atoms.copy()
                    coords[i, : row_coords.shape[0], :] = row_coords.copy()
                except Exception as Ex:
                    # Hack for bad snowflake molecules
                    atoms[i, : row_atoms.shape[0]] = row_atoms.copy()
                    coords[i, : row_coords.shape[0] // 3, :] = row_coords.reshape(
                        (-1, 3), order="C"
                    ).copy()
            else:
                continue
        if return_grads and return_dipole:
            batch.update(
                {
                    "atoms": atoms,
                    "coords": coords,
                    "gradients": grads,
                    "dipoles": dipoles,
                }
            )
        if return_grads:
            batch.update({"atoms": atoms, "coords": coords, "gradients": grads})
        else:
            batch.update({"atoms": atoms, "coords": coords})
    frm = pd.DataFrame(rows)
    # get additional data from dataframe
    for C in frm.columns:
        if C not in batch:
            batch[C] = frm[C].values
    return batch


def get_mod_from_str(x, divisor=100_000):
    return int.from_bytes(hashlib.md5(x.encode("utf-8")).digest(), "little") % divisor


@functional_datapipe("ur_batcher")
class UrBatcher(IterDataPipe):
    def __init__(
        self,
        dp,
        batch_size: int = 32,
        partition: str = "raw",
        xform_routine=lambda X: X,
        partition_routine=lambda X: ["raw", "train", "test"],
        distributed_rankmod_total=None,
        distributed_rankmod_rank=1,
        direct_mode=False,  # for inference.
        required_fields=[],
        skip_last=True,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.partition = partition
        self.partition_routine = partition_routine
        self.xform_routine = xform_routine
        self.required_fields = required_fields
        self.distributed_rankmod_total = (
            distributed_rankmod_total  # for distributed training.
        )
        self.distributed_rankmod_rank = distributed_rankmod_rank
        self.direct_mode = direct_mode
        self.dp = dp
        self.skip_last = skip_last

    def __iter__(self):
        batch = []
        for row in self.dp:
            if not all([key in row.keys() for key in self.required_fields]):
                continue

            row["mod_molecule"] = get_mod_from_str(
                row["smiles"], 100_000
            )  # Now divisble by 100,000!
            # achieves process separation.
            if not self.distributed_rankmod_total is None:
                if not (
                    (row["mod_molecule"] % (self.distributed_rankmod_total))
                    == self.distributed_rankmod_rank
                ):
                    continue
            if not self.partition in self.partition_routine(row):
                continue

            batch.append(row)
            if len(batch) == self.batch_size:
                sX = stack_batch(batch, return_coords=True)
                yield self.xform_routine(sX)
                batch = []
        if len(batch) and not self.skip_last:
            sX = stack_batch(batch, return_coords=True)
            yield self.xform_routine(sX)


@functional_datapipe("unstack_pickles")
class UnstackPickles(IterDataPipe):
    def __init__(self, dp) -> None:
        super().__init__()
        self.dp = dp

    def __iter__(self):
        for path, file in self.dp:
            raw_rows = pickle.load(file)
            yield raw_rows
