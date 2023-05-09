from functools import partial
from multiprocessing import Pool


def calculate_conservation(column, matrix):
    """ Calculate conservation score from alignment result and score matrix.

    Args:
        residues (list, int): (one column of the alignment string in Stockholm format,
            column index).
        matrix (dict): the matrix for conservation score caculation. Keys should be
            symbol pairs in a tuple. Values should be the conservation score for that
            pair.
        id (str): for multiprocessing. To identify the scores.

    Returns (int, str) or (int):
        Conservation score, id or conservation score only if id is None.
    """
    residues, id = column
    score = 0
    for j, res1 in enumerate(residues):
        for k, res2 in enumerate(residues):
            if j <= k:
                continue
            score += matrix[(res1.upper(), res2.upper())]
    if id is not None:
        return score, id
    else:
        return score


def select_positions_by_subcluster_consensus(seqlist, matrix, max_seq_len, ncpu=1):
    """ Select conserved postions based on multi-sequence alignment.
    Args:
        seqlist (list): ['ac-de','a--de','accde','ad-de','accee'].
        max_seq_len (int): maximum length of selected postions.
        ncpu (int): number of cpu to use
    output (list):
        Chosen positions, e.g., [0,3,4].
    """
    import numpy as np

    consensus_scores = [0.0] * len(seqlist[0])

    inputs = []  # list of (column_residues, i) where i is the column index
    seqlist_clear = []
    for seq in seqlist:
        if len(seq) < len(seqlist[0]):
            continue
        else:
            seqlist_clear.append(seq)
    seqlist = seqlist_clear
    for i in range(len(seqlist[0])):
        column_residues = [seq[i] for seq in seqlist]
        inputs.append((column_residues, i))
    if ncpu > 1:
        outputs = []
        cc = partial(calculate_conservation, matrix=matrix)
        with Pool(ncpu) as pool:
            for res in pool.imap_unordered(cc, inputs):
                outputs.append(res)
        for output in outputs:
            consensus_scores[output[1]] = output[0]
    elif ncpu == 1:
        for inp in inputs:
            s, i = calculate_conservation(inp, matrix)
            consensus_scores[i] = s
    else:
        raise ValueError(f"Number of cpus {ncpu} is illegal.")
    selected_positions = sorted(np.argsort(consensus_scores)[::-1][:max_seq_len])
    return selected_positions


def select_positions_by_default_consensus(consensus, max_seq_len):
    """ Select conserved positions based on the "#=GC seq_cons" line in .aln files.

    Args:
        consensus (str): the consensus sequence.
        max_seq_len (int): maximum length of selected postions.

    Return (list):
        Selected postions as list of ints.
    """
    hcpositions = []  # high-conserved
    lcpositions = []  # low-conserved
    cspositions = []  # conservative-substitution
    inspositions = []  # insertion '.'
    delpositions = []  # deletion '-'
    for i, aa in enumerate(consensus):
        if aa.isupper():
            hcpositions.append(i)
        elif aa.islower():
            lcpositions.append(i)
        elif aa == "+":
            cspositions.append(i)
        elif aa == ".":
            inspositions.append(i)
        elif aa == "-":
            delpositions.append(i)
        else:
            continue
    selected_positions = (
        hcpositions + lcpositions + cspositions + delpositions + inspositions
    )
    selected_positions = selected_positions[:max_seq_len]
    selected_positions = sorted(selected_positions)
    return selected_positions
