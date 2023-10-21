import datetime
import gc
import glob
import os
import sys
import json
import multiprocessing as mp
import signal
import shutil
from itertools import product as tensor_product
from datetime import timezone

import torch
import numpy as np


def dir_or_file_exists(d):
    return os.path.exists(d)


def tensor_of_dict_of_lists(d: dict):
    tore_values = list(tensor_product(*d.values()))
    return [{K: V[i] for i, K in enumerate(d.keys())} for V in tore_values]


def colored_background(r: int, g: int, b: int, text):
    """
    r,g,b integers between 0,255
    """
    return f"\033[48;2;{r};{g};{b}m{text}\033[0m"


def batch_indexable(iterable, n=128):
    """
    Simple batching iterator over an iterable that is indexable.
    """

    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx : min(ndx + n, l)]


class NpEncoder(json.JSONEncoder):
    """
    A useful thing to make dicts json compatible.
    """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray) or isinstance(obj, torch.Tensor):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def json_valid_dict(obj):
    return json.loads(json.dumps(obj, cls=NpEncoder))


def utc_epoch_now():
    return datetime.datetime.now().replace(tzinfo=timezone.utc).timestamp()


def makedir(path: str, isfile: bool = False):
    """
    Creates a directory given a path to either a directory or file.
    If a directory is provided, creates that directory. If a file is provided (i.e. isfile == True),
    creates the parent directory for that file.
    :param path: Path to a directory or file.
    :param isfile: Whether the provided path is a directory or file.
    """
    if isfile:
        path = os.path.dirname(path)
    if path != "":
        os.makedirs(path, exist_ok=True)


def rmdir(path: str):
    """
    Creates a directory given a path to either a directory or file.
    If a directory is provided, creates that directory. If a file is provided (i.e. isfile == True),
    creates the parent directory for that file.
    :param path: Path to a directory or file.
    :param isfile: Whether the provided path is a directory or file.
    """
    try:
        shutil.rmtree(path)
    except Exception as Ex:
        print("rmdir failure", Ex)


class OnlineEstimator:
    """
    Simple storage-less Knuth estimator which
    accumulates mean and variance.
    """

    def __init__(self, x_):
        self.n = 1
        self.mean = x_ * 0.0
        self.m2 = x_ * 0.0
        delta = x_ - self.mean
        self.mean += delta / self.n
        delta2 = x_ - self.mean
        self.m2 += delta * delta2

    def __call__(self, x_):
        self.n += 1
        delta = x_ - self.mean
        self.mean += delta / self.n
        delta2 = x_ - self.mean
        self.m2 += delta * delta2
        return self.mean, self.m2 / (self.n - 1)


# useful for debugging Torch memory leaks.
def get_all_allocated_torch_tensors():
    objs = []
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (
                hasattr(obj, "data") and torch.is_tensor(obj.data)
            ):
                objs.append(obj)
        except:
            pass
    return objs


def records_mp(recs, func, args=None, n=None):
    """Apply func(chunk_recs, *args) to chunks of input records using multiprocessing"""
    if n is None:
        n = min([mp.cpu_count(), len(recs)])
    if args is None:
        args = tuple()

    before_ct = len(recs)
    mp_args = [(sub_recs, *args) for sub_recs in batch_indexable(recs, n)]
    with mp.Pool(processes=n) as pool:
        recs = pool.starmap(func, mp_args)

    recs = [rec for sub_recs in recs for rec in sub_recs]
    assert len(recs) == before_ct

    return recs


def execute_with_timeout(method, args, timeout):
    """Execute method with timeout, return None if timeout exceeded"""
    result = None

    def timeout_handler(signum, frame):
        # This function is called when the timeout is reached
        # It raises an exception to stop the execution of the method
        raise TimeoutError("Execution timed out")

    # Set up the timeout signal handler
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)  # Start the timeout timer

    try:
        result = method(*args)  # Execute the method
    except TimeoutError:
        pass  # Execution timed out
    finally:
        signal.alarm(0)  # Cancel the timeout timer

    return result


def get_tnet_dir():
    return os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


def dicts_to_keyval(list_of_dicts, key: str, value: str):
    # convert list of dictionaries with unique keys to key value mapping.
    return {dct[key]: dct[value] for dct in list_of_dicts}


def query_yes_no(question, default=None):
    """https://stackoverflow.com/questions/3041986/apt-command-line-interface-like-yes-no-input
    Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
            It must be "yes" (the default), "no" or None (meaning
            an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == "":
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' " "(or 'y' or 'n').\n")
