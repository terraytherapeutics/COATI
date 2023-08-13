import numpy as np

# this isn't really used anymore.


def adj_mat_to_tokens(
    adj_mat: np.ndarray,
    adj_mat_atoms: np.ndarray,
    only_heavy: bool = True,
):
    """
    This is basically the TokenGT paper
    the reverse is decode_tokenized_graph()
    """
    if np.isnan(adj_mat_atoms).any():
        return ""
    elif adj_mat_atoms[adj_mat_atoms > 1].shape[0] > 150:
        return ""
    try:
        atom_string = ""
        edge_string = ""

        light_to_heavy = np.zeros(adj_mat_atoms.shape[0], dtype=int)
        light_to_heavy[adj_mat_atoms > 1] = np.arange(
            (adj_mat_atoms > 1).sum(), dtype=int
        )

        for I, at in enumerate([adj_mat_atoms[K] for K in range(len(adj_mat_atoms))]):
            if only_heavy:
                if at < 2:
                    continue
            atom_string = (
                atom_string
                + "[NUM"
                + str(light_to_heavy[I])
                + "]"
                + "[ELM"
                + str(int(at))
                + "]"
            )
        for I, ed in enumerate(adj_mat):
            if only_heavy:
                if adj_mat_atoms[int(ed[0])] < 2:
                    continue
                if adj_mat_atoms[int(ed[1])] < 2:
                    continue
            if float(ed[2]) == 1:
                et = "[EDGE1]"
            elif float(ed[2]) > 1 and float(ed[2]) < 2:
                et = "[EDGEC]"
            elif int(ed[2]) == 2:
                et = "[EDGE2]"
            elif int(ed[2]) == 3:
                et = "[EDGE3]"
            else:
                et = "[EDGE0]"

            # The edge node numbers should always be sorted.
            edge_numbers = sorted(
                [light_to_heavy[int(ed[0])], light_to_heavy[int(ed[1])]]
            )
            edge_string = (
                edge_string
                + et
                + "[NUM"
                + str(edge_numbers[0])
                + "]"
                + "[NUM"
                + str(edge_numbers[1])
                + "]"
            )
        return "[GRAPH]" + atom_string + "[EDGES]" + edge_string
    except Exception as Ex:
        print(Ex)
        raise Ex
