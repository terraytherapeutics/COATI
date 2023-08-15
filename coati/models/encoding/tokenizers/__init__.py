import json
import os
from pathlib import Path
from typing import Dict, List

from coati.common.s3 import cache_read

from .smiles_vocab import tokenizer_vocabs

# absolute path to the vocabulary folder
VOCAB_PATH = Path(__file__).parent / "vocabs"


def load_vocab(vocab_name: str) -> Dict[str, List[str]]:
    with open(VOCAB_PATH / f"{vocab_name}.json", "r") as f:
        return json.load(f)


def get_vocab(vocab_name: str) -> Dict[str, List[str]]:
    try:
        return tokenizer_vocabs[vocab_name]
    except KeyError:
        print("vocab_name not found in tokenizer_vocabs, trying to load from file")

    try:
        return load_vocab(vocab_name)
    except:
        raise ValueError(f"vocab_name {vocab_name} not found in vocabs folder")
