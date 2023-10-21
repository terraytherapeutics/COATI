from typing import Any, Dict, List, Union
from coati.common.util import makedir, utc_epoch_now
from contextlib import contextmanager
import pickle
import socket
import json
import os


class COATILogger:
    """
    Basic training / artifact logger that just caches desired stuff to mongo.
    for later queries.
    Separated this out from the other classes designed for training/test data retrieval.

    Ex:
    logger = tnet_logger(model_name = 'xgb', args = vars(args))
    logger.start()
    for batch in epoch:
        ...
        logger.log_metric('train_loss', train_loss)

    logger.stop()
    """

    def __init__(
        self,
        model_name: str,
        output_path: str,
        model_path: str,
        tags: List[str] = [],
        run_time=None,
        dataset: str = None,
        args: Dict[str, Any] = None,
    ):
        self._running = False
        self.output_path = output_path
        self.model_path = model_path
        self.model_name = model_name
        self.tags = tags
        self.run_time = run_time
        self.args = args
        self.dataset = dataset

    def start(self):
        self._running = True
        if self.run_time is None:
            self.run_time = str(int(utc_epoch_now()))
        self.run_host = socket.gethostname()
        makedir(os.path.join(self.output_path, self.run_time))
        self.log_file = os.path.join(self.output_path, self.run_time, "log.json")
        with open(self.log_file, "w") as f:
            f.write("[")
        return

    def stop(self, failed: bool = False, details: str = None):
        with open(self.log_file, "a") as f:
            f.write("]")
        return

    def log_metric(
        self,
        key: str,
        value: Any,
        dataset_epoch: int = None,
        step: int = None,
        tags: Dict[str, str] = None,
    ):
        to_insert = {
            "event": "metric",
            "epoch": str(int(utc_epoch_now())),
            "run_time": self.run_time,
            "model_name": self.model_name,
            "key": key,
            "value": value,
        }
        if not dataset_epoch is None:
            to_insert["dataset_epoch"] = dataset_epoch
        if not step is None:
            to_insert["step"] = step
        if tags is not None:
            to_insert.update(
                {"tag_" + tag_key: tag_val for tag_key, tag_val in tags.items()}
            )

        with open(self.log_file, "a") as f:
            f.write(json.dumps(to_insert) + ",")

        return to_insert

    def log_metrics(self, metrics: Dict[str, Any], **kwargs):
        """Simple unpacking of a dictionary. Enables passing of bulk metrics.

        Args:
            metrics (Dict[str, Any]): Metrics dictionary
            kwargs: i.e. dataset_epoch:epoch...
        """
        # one-liner map(self.log_metric, metrics.items()) JAP
        for key, val in metrics.items():
            self.log_metric(key, val, **kwargs)

    def log_epoch_stats(self, epoch_stats, tags=None):
        if not self._running:
            raise RuntimeError("Please save predictions before ending the run.")
        epoch = str(int(utc_epoch_now()))
        epoch_suffix = f"epoch_stats/{self.model_name}_{self.run_time}_{epoch}"
        epoch_url = os.path.join(self.output_path, epoch_suffix)
        to_insert = {
            "event": "epoch_stats",
            "epoch": epoch,
            "run_time": self.run_time,
            "model_name": self.model_name,
            "document": epoch_url,
        }
        if tags is not None:
            to_insert.update(
                {"tag_" + tag_key: tag_val for tag_key, tag_val in tags.items()}
            )
        # need to do this for proper multipart uploading.
        with open(self.log_file, "a") as f:
            f.write(json.dumps(to_insert) + ",")
        return

    def get_model_path(self, name, run_time, epoch):
        return os.path.join(self.model_path, f"{name}_{run_time}_{epoch}")

    def _save_model_artifact(self, artifact):
        epoch = str(int(utc_epoch_now()))
        model_url = self.get_model_path(
            name=self.model_name, run_time=self.run_time, epoch=epoch
        )
        with open(model_url, "wb") as f_out:
            f_out.write(artifact)
        return model_url, epoch

    def _log_model_artifact(
        self, artifact: Any, model_event: str, tags: Dict[str, str] = None
    ):
        model_url, epoch = self._save_model_artifact(artifact)

        print("Logged Artifact to:", model_url)

        return

    def log_pytorch(self, model_document, tags: Dict[str, str] = None):
        print(f"Logging model run_time {self.run_time}")
        return self._log_model_artifact(
            model_document, model_event="pytorch_model", tags=tags
        )


@contextmanager
def coati_logger(
    model_name: str,
    output_path: str,
    model_path: str,
    tags: List[str] = [],
    run_time=None,
    dataset: str = None,
    args: Dict[str, Any] = None,
):
    logger = coati_logger(
        model_name, output_path, model_path, tags, run_time, dataset, args
    )
    logger.start()
    try:
        yield logger
    except Exception as e:
        logger.stop(failed=True, details=str(e))
        raise e
    else:
        logger.stop()
