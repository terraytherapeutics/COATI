"""
loads data used for training COATI.

c.f. make_cache. which does a lot of aggs. 
"""
import os

from torch.utils.data.datapipes.iter import FileLister, Shuffler

from coati.common.util import dir_or_file_exists, makedir, query_yes_no
from coati.common.s3 import copy_bucket_dir_from_s3
from coati.data.batch_pipe import UnstackPickles, UrBatcher, stack_batch


S3_PATH = "datasets/coati_data/"


class COATI_dataset:
    def __init__(
        self,
        cache_dir,
        fields=["smiles", "atoms", "coords"],
        test_split_mode="row",
        test_frac=0.02,  # in percent.
        valid_frac=0.02,  # in percent.
    ):
        self.cache_dir = cache_dir
        self.summary = {"dataset_type": "coati", "fields": fields}
        self.test_frac = test_frac
        self.fields = fields
        self.valid_frac = valid_frac
        assert int(test_frac * 100) >= 0 and int(test_frac * 100) <= 50
        assert int(valid_frac * 100) >= 0 and int(valid_frac * 100) <= 50
        assert int(valid_frac * 100 + test_frac * 100) < 50
        self.test_split_mode = test_split_mode

    def partition_routine(self, row):
        """ """
        if not "mod_molecule" in row:
            tore = ["raw"]
            tore.append("train")
            return tore
        else:
            tore = ["raw"]

            if row["mod_molecule"] % 100 >= int(
                (self.test_frac + self.valid_frac) * 100
            ):
                tore.append("train")
            elif row["mod_molecule"] % 100 >= int((self.test_frac * 100)):
                tore.append("valid")
            else:
                tore.append("test")

            return tore

    def get_data_pipe(
        self,
        rebuild=False,
        batch_size=32,
        partition: str = "raw",
        required_fields=[],
        distributed_rankmod_total=None,
        distributed_rankmod_rank=1,
        xform_routine=lambda X: X,
    ):
        """
        Look for the cache locally
        then on s3 if it's not available locally
        then return a pipe to the data.
        """
        print(f"trying to open a {partition} datapipe for...")
        if (
            not dir_or_file_exists(os.path.join(self.cache_dir, S3_PATH, "0.pkl"))
        ) or rebuild:
            makedir(self.cache_dir)
            query_yes_no(
                f"Will download ~340 GB of data to {self.cache_dir} . This will take a while. Are you sure?"
            )
            copy_bucket_dir_from_s3(S3_PATH, self.cache_dir)

        pipe = (
            FileLister(
                root=os.path.join(self.cache_dir, S3_PATH),
                recursive=False,
                masks=["*.pkl"],
            )
            .shuffle()
            .open_files(mode="rb")
            .unstack_pickles()
            .unbatch()
            .shuffle(buffer_size=200000)
        )
        pipe = pipe.ur_batcher(
            batch_size=batch_size,
            partition=partition,
            xform_routine=xform_routine,
            partition_routine=self.partition_routine,
            distributed_rankmod_total=distributed_rankmod_total,
            distributed_rankmod_rank=distributed_rankmod_rank,
            direct_mode=False,
            required_fields=self.fields,
        )
        return pipe
