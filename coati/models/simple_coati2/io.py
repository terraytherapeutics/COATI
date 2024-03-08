# model loading function for a molclip model.
from io import BytesIO
import pickle

import torch

from coati.common.s3 import cache_read
from coati.models.simple_coati2.transformer_only import COATI_Smiles_Inference
from coati.models.simple_coati2.trie_tokenizer import TrieTokenizer
from coati.models.encoding.tokenizers import get_vocab


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(BytesIO(b), map_location="cpu")
        else:
            return super().find_class(module, name)


def load_coati2(
    doc_url: str,
    device: str = "cpu",
    freeze: bool = True,
    old_architecture=False,
    force_cpu=False,  # needed to deserialize on some cpu-only machines
):

    print(f"Loading model from {doc_url}")

    with cache_read(doc_url, "rb") as f_in:
        if force_cpu:
            model_doc = CPU_Unpickler(f_in, encoding="UTF-8").load()
        else:
            model_doc = pickle.loads(f_in.read(), encoding="UTF-8")
    model_kwargs = model_doc["model_kwargs"]

    model_dict_ = model_doc["model"]
    new_names = [
        k.replace("module.", "") if k.startswith("module.") else k
        for k in model_dict_.keys()
    ]
    state_dict = {new_name: t for new_name, t in zip(new_names, model_dict_.values())}

    tokenizer_vocab = model_doc["train_args"]["tokenizer_vocab"]
    print(f"Loading tokenizer {tokenizer_vocab} from {doc_url}")

    if old_architecture:
        model_kwargs["old_architecture"] = True

    if "device" in model_kwargs:
        model_kwargs["device"] = device

    # Let's just be explicit for our use case these are the values for the reduced model
    updated_kwargs = {
        "n_layer_xformer": model_kwargs["n_layer_xformer"],
        "n_hidden_xformer": model_kwargs["n_hidden_xformer"],
        "embed_dim": model_kwargs["embed_dim"],
        "n_head": model_kwargs["n_head"],
        "n_seq": model_kwargs["n_seq"],
        "mlp_dropout": model_kwargs["mlp_dropout"],
        "enc_to_coati": model_kwargs["enc_to_coati"],
        "n_direct_clr": model_kwargs["n_direct_clr"],
        "n_tok": model_kwargs["n_tok"],
        "biases": model_kwargs["biases"],
        "device": model_kwargs["device"],
        "dtype": model_kwargs["dtype"],
    }

    model = COATI_Smiles_Inference(**updated_kwargs)

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.device = device
    tokenizer = TrieTokenizer(n_seq=model_kwargs["n_seq"], **get_vocab(tokenizer_vocab))

    if freeze:
        print("Freezing encoder")
        n_params = 0
        for param in model.parameters():
            param.requires_grad = False
            n_params += param.numel()
        print(f"{n_params } params frozen!")
    return model, tokenizer
