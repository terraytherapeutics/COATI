from coati.common.util import colored_background
from coati.models.encoding.tokenizers.trie import Trie
import torch
from typing import Tuple, List


class TrieTokenizer:
    """
    Converts smiles+sentinel tokens into a list of integers.
    """

    def __init__(
        self,
        n_seq=256,  # The dimension of the token embedding.
        smiles_tokens=[],
        special_tokens=[],
        side_tasks=True,
    ):
        self.n_seq = n_seq
        self.special_tokens = special_tokens
        self.smiles_tokens = smiles_tokens
        self.keys = self.special_tokens + self.smiles_tokens
        self.n_token = len(self.keys)  # base number of tokens.
        self.vocab = {T.strip(): I for I, T in enumerate(self.keys)}

        # I am human, after all.
        # These are tokens wrt, the model should be uniform (loss masked)
        self.stop_token = self.vocab["[STOP]"]
        self.pad_token = self.vocab["[PAD]"]

        self.clip_token = self.vocab["[CLIP]"]
        self.unk_token = self.vocab["[UNK]"]
        self.smiles_token = self.vocab["[SMILES]"]
        self.suffix_token = self.vocab["[SUFFIX]"]
        self.middle_token = self.vocab["[MIDDLE]"]
        if side_tasks:
            self.graph_token = self.vocab["[GRAPH]"]
            self.formula_token = self.vocab["[FORMULA]"]
            self.set_token = self.vocab["[SET]"]

        self.smiles_trie = Trie()
        self.special_trie = Trie()
        for k in self.special_tokens:
            self.special_trie.add(k)
        for k in self.smiles_tokens:
            self.smiles_trie.add(k)

    def pre_tokenize(self, text):
        """
        Splits the special tokens first.
        """
        split0 = self.special_trie.split(text)
        tokens = []
        for T in split0:
            if T in self.special_tokens:
                tokens.append(T)
            else:
                tokens.extend(self.smiles_trie.split(T))
        return tokens

    def tokenize_text(
        self, text: str, pad: bool = True, range_check: bool = True
    ) -> List[int]:
        """
        Tokenizes a single row.
        """
        try:
            tore = [self.vocab[T] for T in self.pre_tokenize(text)]
            if len(tore) > self.n_seq and range_check:
                raise Exception("Oversized String", len(tore))
            if pad:
                tore = tore + [
                    self.vocab["[PAD]"] for k in range(self.n_seq - len(tore))
                ]
        except Exception as Ex:
            print("tokenize text exception... ", text, Ex, self.pre_tokenize(text))
            raise Ex
        return tore

    def batch_smiles(
        self, smiles_batch: List[str], device: str = "cpu", skip_failed: bool = False
    ) -> Tuple[torch.Tensor, List[int]]:
        token_stack = []
        bad_idxs = []
        for idx, smi in enumerate(smiles_batch):
            try:
                ttext = self.tokenize_text(
                    "[SMILES]" + smi + "[STOP]", pad=False, range_check=False
                )
            except KeyError as e:
                if skip_failed:  # filling with a dummy string, and adding to bad_idxs
                    ttext = self.tokenize_text(
                        "[SMILES]" + "C" + "[STOP]", pad=False, range_check=False
                    )
                    bad_idxs.append(idx)
                else:
                    raise e

            if len(ttext) <= self.n_seq:
                t = torch.zeros(self.n_seq, dtype=torch.long, device=device)
                t[: len(ttext)] = torch.tensor(ttext)
                token_stack.append(t)
            else:
                bad_idxs.append(idx)

        new_smi_batch = torch.stack(token_stack, 0)
        new_smi_batch = new_smi_batch[:, : (new_smi_batch.sum(0) > 0).sum()]
        return new_smi_batch, bad_idxs

    def decode(
        self,
        ints,
        special=True,
        end_at_stop=True,
        de_fim=True,
        color_loss=None,  # Provides colored likelihoods in blue
    ):
        """
        Detokenizes a single row.

        Args:
            ints: a list of token integers
            special: decode special tokens? (if False they are mapped to '')
            de_fim: undo fill-in-middle
        Returns:
            a string of decoded tokens.
        """
        if not len(ints):
            return ""
        assert type(ints[0]) == int
        if end_at_stop and self.stop_token in ints:
            ints = ints[: ints.index(self.stop_token) + 1]

        if not color_loss is None:
            assert len(color_loss) >= len(ints)
            max_loss = max(color_loss)
            min_loss = min(color_loss)
            strings = [
                colored_background(
                    int((color_loss[i] - min_loss) / (max_loss - min_loss) * 255),
                    128,
                    128,
                    self.keys[I],
                )
                for i, I in enumerate(ints)
                if I > 0
            ]
        else:
            strings = [self.keys[I] for I in ints if I > 0]

        if special:
            if de_fim and "[MIDDLE]" in strings and "[SUFFIX]" in strings:
                si = strings.index("[SUFFIX]")
                mi = strings.index("[MIDDLE]")
                return "".join(
                    strings[:si] + strings[mi:-1] + strings[si:mi] + strings[-1:]
                )
            else:
                return "".join(strings)
        else:
            if de_fim and "[MIDDLE]" in strings and "[SUFFIX]" in strings:
                si = strings.index("[SUFFIX]")
                mi = strings.index("[MIDDLE]")
                ordd = strings[:si] + strings[mi:-1] + strings[si:mi] + strings[-1:]
                return "".join([S for S in ordd if not S in self.special_tokens])
            else:
                return "".join([S for S in strings if not S in self.special_tokens])
