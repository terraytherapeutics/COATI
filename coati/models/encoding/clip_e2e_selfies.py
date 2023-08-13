#
# This version does clip only on naked smiles strings.
# and does smiles => smiles e2e.
#

import random
import numpy as np

import torch
from coati.models.encoding.fill_in_middle import adj_mat_to_tokens


def selfies_pre_tokenize(self, text):
    """
    Splits the special tokens first.
    """
    import selfies as sf

    split0 = self.special_trie.split(text)
    tokens = []
    for T in split0:
        if T in self.special_tokens:
            tokens.append(T)
        else:
            tokens.extend(self.smiles_trie.split(sf.encoder(T)))
    return tokens


def to_selfies_tokenizer(tokenizer):
    tokenizer.pre_tokenize = selfies_pre_tokenize.__get__(tokenizer)
    return tokenizer


def clip_ar_xform_selfies(
    batch,
    tokenizer,
    p_dataset=0.2,
    p_formula=0.2,
    p_fim=0.0,
    p_graph=0.0,
    p_clip=0.9,
    p_clip_cut=0.3,
    p_randsmiles=0.0,  # NOTE: this is applied BEFORE raw_tokens are collected if 0.
    dtype=torch.float,
    device=torch.device("cpu"),
    coord_noise=False,
):
    """
    This verison randomly augments data in several ways (front or back)
    it randomly prepends the dataset, it randomly prepends the
    molecular formula and randomly performs the prefix, suffix, middle

    Note: this exploits the fact that the encoding will always begin with [SMILES]
    and end with [STOP] [PAD] [PAD] ... and that [PAD] is always token mapping to 0

    Here's examples.
    smiles: "c1ccccc1"
    No augmentations: [SMILES][c1][cccc][c1][STOP]

    dataset aug: [SET][tensormol][SMILES][c1][cccc][c1][STOP] or
                 [SMILES][c1][cccc][c1][SET][tensormol][STOP]
    formula aug: [FORMULA][ELM6][NUM6][ELM1][NUM6][SMILES][c1][cccc][c1][STOP] or
                 [SMILES][c1][cccc][c1][FORMULA][ELM6][NUM6][ELEMENT][ELM1][NUM6][STOP] or
    partialform: [ELM6][NUM6][SMILES][c1][cccc][c1][STOP]
    graph aug  : [GRAPH][NODE][ELM1][NUM1][ELM1][NUM2][EDGE1][NUM1][NUM2][BOND1]
    parital    : [NODE][ELM1][NUM1][NODE][ELM1][NUM2][EDGE1][NUM1][NUM2][BOND1]

    Fill-in-middle is always applied AFTER previous possible augmentations
    so dataset can be filled in if desired but not on the string level
    so these special tokens aren't broken.

    fim aug:
    [ELEMENT][Tk6][Tk6][ELEMENT][Tk1][Tk6][SMILES][c1][cccc][c1][STOP] becomes:
    [PREFIX][ELEMENT][Tk6][Tk6][ELEMENT][Tk1][Tk6][SMILES][c1][SUFFIX][c1][MIDDLE][cccc]
    """
    CANON_TEXT = "selfies"  # selfies already canonical in cache
    RAND_TEXT = "rand_selfies"  # rand_selfies already permuted in cache
    assert CANON_TEXT in batch
    assert "source_collection" in batch
    assert "atoms" in batch
    assert "coords" in batch

    token_stack = []
    s2s_stack = []
    for k, S_ in enumerate(batch[CANON_TEXT]):
        try:
            # special token will still be [SMILES]
            reps = ["smiles"]

            # Choose augmentations
            if random.random() < p_dataset:
                if (
                    "[" + batch["source_collection"][k] + "]"
                    in tokenizer.special_tokens
                ):
                    reps.append("set")
            if random.random() < p_formula:
                reps.append("formula")
            if (
                random.random() < p_graph
                and "adj_mat" in batch
                and "adj_mat_atoms" in batch
            ):
                reps.append("graph")

            random.shuffle(reps)
            S = ""
            for rep in reps:
                if rep == "set":
                    S = S + "[SET]" + "[" + batch["source_collection"][k] + "]"
                elif rep == "smiles":
                    S = S + "[SMILES]" + S_
                elif rep == "formula":
                    ats = batch["atoms"][k].astype(int)
                    cts = np.bincount(ats[ats > 0])
                    if (cts < 150).all():
                        rows = np.stack(
                            [np.arange(0, cts.shape[0])[cts > 0], cts[cts > 0]], -1
                        )
                        formula_string = "[FORMULA]" + "".join(
                            [
                                "[ELM" + str(r[0]) + "][NUM" + str(r[1]) + "]"
                                for r in rows
                            ]
                        )
                    else:
                        formula_string = ""
                    S = S + formula_string
                elif rep == "graph":
                    graph_string = adj_mat_to_tokens(
                        batch["adj_mat"][k], batch["adj_mat_atoms"][k]
                    )
                    S = S + graph_string

            S = S + "[STOP]"
            ttext = tokenizer.tokenize_text(S, pad=False, range_check=False)

            if random.random() < p_clip and len(ttext) > 3:
                if random.random() < p_clip_cut:
                    # CLIP Augmentation (hidden token always goes to position [1])
                    # Pull out the stop token.
                    # try [CLIP][UNK][SMILES][SUFFIX][MIDDLE].....
                    stop_token = ttext.pop()
                    # choose the positions of [MIDDLE] and [SUFFIX]
                    # note: they cannot be the first two tokens.
                    mp, sp = 1, 1
                    while mp == sp:  #  or mp > ttext_ind_smiles:
                        mp, sp = sorted(
                            [
                                random.randint(2, len(ttext)),
                                random.randint(2, len(ttext)),
                            ]
                        )
                    ttext = (
                        tokenizer.tokenize_text(
                            "[CLIP][UNK]", pad=False, range_check=False
                        )
                        + ttext[:mp]
                        + tokenizer.tokenize_text(
                            "[SUFFIX]", pad=False, range_check=False
                        )
                        + ttext[sp:]
                        + tokenizer.tokenize_text(
                            "[MIDDLE]", pad=False, range_check=False
                        )
                        + ttext[mp:sp]
                        + [stop_token]
                    )
                else:
                    ttext = (
                        tokenizer.tokenize_text(
                            "[CLIP][UNK]", pad=False, range_check=False
                        )
                        + ttext
                    )
            elif random.random() < p_fim and len(ttext) > 4:
                # Fill-in-middle Augmentation
                # Pull out the stop token.
                stop_token = ttext.pop()
                # choose the positions of [MIDDLE] and [SUFFIX]
                # note: they cannot be the first token.
                mp, sp = 1, 1
                while mp == sp:
                    mp, sp = sorted(
                        [random.randint(1, len(ttext)), random.randint(1, len(ttext))]
                    )
                ttext = (
                    tokenizer.tokenize_text("[PREFIX]", pad=False, range_check=False)
                    + ttext[:mp]
                    + tokenizer.tokenize_text("[SUFFIX]", pad=False, range_check=False)
                    + ttext[sp:]
                    + tokenizer.tokenize_text("[MIDDLE]", pad=False, range_check=False)
                    + ttext[mp:sp]
                    + [stop_token]
                )

            if random.random() < p_randsmiles:
                S_raw = "[SMILES]" + batch[RAND_TEXT][k] + "[STOP]"
                s2s_text = tokenizer.tokenize_text(S_raw, pad=False, range_check=False)
                unnperm_toks = tokenizer.tokenize_text(
                    "[SMILES]" + S_ + "[STOP]", pad=False, range_check=False
                )
            else:
                S_raw = "[SMILES]" + S_ + "[STOP]"
                s2s_text = tokenizer.tokenize_text(S_raw, pad=False, range_check=False)
                unnperm_toks = s2s_text

            if len(ttext) <= tokenizer.n_seq and len(s2s_text) <= tokenizer.n_seq:
                t = torch.zeros(tokenizer.n_seq, dtype=torch.long, device=device)
                t[: len(ttext)] = torch.tensor(ttext)

                smi_t = torch.zeros(tokenizer.n_seq, dtype=torch.long, device=device)
                smi_t[: len(s2s_text)] = torch.tensor(s2s_text)

                token_stack.append(t)
                s2s_stack.append(smi_t)
            else:
                # try to just make it a simple smiles if it got oversized.
                # But still always canonically decode (token stack gets unpermed)
                if (
                    len(s2s_text) <= tokenizer.n_seq
                    and len(unnperm_toks) <= tokenizer.n_seq
                ):
                    t = torch.zeros(tokenizer.n_seq, dtype=torch.long, device=device)
                    t[: len(unnperm_toks)] = torch.tensor(unnperm_toks)

                    smi_t = torch.zeros(
                        tokenizer.n_seq, dtype=torch.long, device=device
                    )
                    smi_t[: len(s2s_text)] = torch.tensor(s2s_text)

                    token_stack.append(t)
                    s2s_stack.append(smi_t)
                else:
                    s2s_stack.append(
                        torch.cat(
                            [
                                tokenizer.stop_token
                                * torch.ones(1, dtype=torch.long, device=device),
                                torch.zeros(
                                    tokenizer.n_seq - 1, dtype=torch.long, device=device
                                ),
                            ],
                            0,
                        )
                    )
                    token_stack.append(
                        torch.zeros(tokenizer.n_seq, dtype=torch.long, device=device)
                    )
                    print("Too much seq data.", S_raw, len(s2s_text))
                    continue

        except Exception as Ex:
            print("Tokenize failure:", S_, " Except:", Ex)
            # raise Ex
            s2s_stack.append(
                torch.cat(
                    [
                        tokenizer.stop_token
                        * torch.ones(1, dtype=torch.long, device=device),
                        torch.zeros(
                            tokenizer.n_seq - 1, dtype=torch.long, device=device
                        ),
                    ],
                    0,
                )
            )
            token_stack.append(
                torch.zeros(tokenizer.n_seq, dtype=torch.long, device=device)
            )
            continue

    batch["tokens"] = torch.stack(token_stack, 0)
    batch["raw_tokens"] = torch.stack(s2s_stack, 0)

    if batch["atoms"].shape[0] < 1:
        raise Exception("empty batch")

    for col in ["tokens", "atoms", "raw_tokens"]:
        if col in batch:
            if type(batch[col]) != torch.Tensor:
                batch[col] = torch.tensor(batch[col], requires_grad=False).to(
                    device, torch.long
                )
    for col in ["coords"]:
        if col in batch:
            if type(batch[col]) != torch.Tensor:
                batch[col] = torch.tensor(batch[col], requires_grad=False).to(
                    device, dtype
                )
    if coord_noise:
        batch["coords"] += torch.normal(
            torch.zeros_like(batch["coords"]), 0.05 * torch.ones_like(batch["coords"])
        )

    # decrease the sequence size to max demanded by this batch.
    batch["tokens"] = batch["tokens"][:, : (batch["tokens"].sum(0) > 0).sum()]
    batch["raw_tokens"] = batch["raw_tokens"][
        :, : (batch["raw_tokens"].sum(0) > 0).sum()
    ]

    # Alignment in cross entropy:
    # [SMILES],  [TOKEN 1]... [STOP] [PAD]
    # [TOKEN 1] [TOKEN 2]... [PAD]  [PAD]
    batch["y_next"] = torch.zeros_like(batch["tokens"])
    batch["y_next"][:, : (batch["tokens"].shape[1] - 1)] = batch["tokens"][
        :, 1:
    ].clone()
    # Critical! no loss for predictions of the pad
    batch["y_next"][batch["y_next"] == tokenizer.clip_token] = -1
    batch["y_next"][batch["y_next"] == tokenizer.pad_token] = -1
    batch["y_next"][batch["y_next"] == tokenizer.unk_token] = -1
    batch["y_next"][batch["y_next"] == tokenizer.suffix_token] = -1
    batch["y_next"][batch["y_next"] == tokenizer.middle_token] = -1
    return batch
