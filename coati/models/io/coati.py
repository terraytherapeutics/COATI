import pickle
from typing import Tuple
from io import BytesIO

import torch.nn as nn
import torch

from coati.common.s3 import cache_read
from coati.models.encoding.clip_e2e import e3gnn_smiles_clip_e2e
from coati.models.encoding.clip_fp_e2e import e3gnn_smiles_clip_e2e as fp_e2e_model
from coati.models.encoding.clip_e2e_selfies import to_selfies_tokenizer
from coati.models.encoding.tokenizers import get_vocab
from coati.models.encoding.tokenizers.trie_tokenizer import TrieTokenizer



class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(BytesIO(b), map_location="cpu")
        else:
            return super().find_class(module, name)


def load_e3gnn_smiles_clip_e2e(
    doc_url: str,
    device: str = "cpu",
    freeze: bool = True,
    strict: bool = False,
    old_architecture=False,
    override_args=None,  # hopefully not needed, but you never know.
    model_type="default",
    print_debug=False,
) -> Tuple[e3gnn_smiles_clip_e2e, TrieTokenizer]:
    """
    Simple model loading function that loads a model from a pickle file.
    Returns an encoder and a tokenizer.
    """
    print(f"Loading model from {doc_url}")

    with cache_read(doc_url, "rb") as f_in:
        if device == "cpu":
            model_doc = CPU_Unpickler(f_in, encoding="UTF-8").load()
        else:
            model_doc = pickle.loads(f_in.read(), encoding="UTF-8")
    model_kwargs = model_doc["model_kwargs"]

    if print_debug:
        from matplotlib import pyplot as plt

        ar_x = [X["tag_n_toks"] for X in model_doc["offline_loss"]["ar_losses"]]
        ar_y = [X["value"] for X in model_doc["offline_loss"]["ar_losses"]]
        cos_x = [X["tag_n_toks"] for X in model_doc["offline_loss"]["clip_losses"]]
        cos_y = [X["value"] for X in model_doc["offline_loss"]["clip_losses"]]
        plt.plot(ar_x, ar_y, label="ar")
        plt.plot(cos_x, cos_y, label="clip")
        plt.legend()
        plt.xlabel("tokens")
        plt.ylabel("loss")
        plt.show()
        print("NTokens: ", model_doc["n_toks_processed"])
        print("Model kwargs: ", model_kwargs)

    model_dict_ = model_doc["model"]
    # if model was created with DataParallel, remove the "module." prefix
    new_names = [
        k.replace("module.", "") if k.startswith("module.") else k
        for k in model_dict_.keys()
    ]
    state_dict = {new_name: t for new_name, t in zip(new_names, model_dict_.values())}

    tokenizer_vocab = model_doc["train_args"]["tokenizer_vocab"]
    print(f"Loading tokenizer {tokenizer_vocab} from {doc_url}")

    if old_architecture:
        model_kwargs["old_architecture"] = True

    if override_args:
        model_kwargs.update(override_args)
    if model_type == "default":
        model = e3gnn_smiles_clip_e2e(**model_kwargs)
    elif model_type == "fp":
        model = fp_e2e_model(**model_kwargs)
    else:
        raise ValueError("unknown model type")
    model.load_state_dict(state_dict, strict=strict)
    model.to(device)
    model.device = device
    tokenizer = TrieTokenizer(n_seq=model_kwargs["n_seq"], **get_vocab(tokenizer_vocab))
    if "selfies" in tokenizer_vocab:
        tokenizer = to_selfies_tokenizer(tokenizer)

    if freeze:
        print("Freezing encoder")
        n_params = 0
        for param in model.parameters():
            param.requires_grad = False
            n_params += param.numel()
        print(f"{n_params } params frozen!")
    return model, tokenizer


def load_offline_loss(doc_url: str):
    """
    Just loads the loss curve from a pickle file.
    """
    print("Loading Loss from offline training")
    with cache_read(doc_url, "rb") as f_in:
        model_doc = pickle.loads(f_in.read(), encoding="UTF-8")
    offline_loss = model_doc["offline_loss"]
    return offline_loss
