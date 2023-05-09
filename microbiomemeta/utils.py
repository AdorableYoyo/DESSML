import numpy as np
from rdkit import Chem


def load_pairs_from_file(pairfile, sep=",", header=True):
    """ Load compound-protein pairs from file.

    Args:
        pairfile (str): path to the file saving compound-protein pairs.
        sep (str): separator symbol for compound and protein. Default is ",".
        header (bool): if the pairfile contains header line.

    Return (dict): compound-protein pair dictionary with edges (str) as keys and
        compound-protein pair tuple (compound ID, protein ID) as values.
    """
    # default data format:
    # chemical compound ID,protein ID (sep=',')
    # e.g. MAEHEIXUINDDHE-UHFFFAOYSA-N,P48736

    pairs = {}
    with open(pairfile, "r") as f:
        if header:
            next(f)
        for line in f:
            line = line.strip().split(sep)
            chem_id = line[0]
            prot_id = line[1]
            edge = chem_id + "\t" + prot_id
            pairs[edge] = (chem_id, prot_id)
    return pairs


def moving_average(a, n=3):
    """ Moving average calculator.

    Args:
        a (Array): numpy array with value to compute moving averages.
        n (int): number of data points to compute the average.

    Return (array): array of moving average values.
    """
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


def is_valid_smiles(smiles):
    """ Test if the SMILES string can be converted to RDKit Molecule object.

    Args:
        smiles (str): the SMILES string to be tested.

    Return (bool): True if convertable, False otherwise.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    return True


def perf_measure(y_actual, y_hat):
    """ Performance measurement. Count true positives, false positives, true negatives,
    and faulse negatives.

    Args:
        y_actual (Array): true labels.
        y_hat (Array): predictions. Integers.

    Returns:
        TP (int): number of true positives.
        FP (int): number of false positives.
        TN (int): number of true negatives.
        FN (int): number of false negatives.
    """
    TP, FP, TN, FN = 0, 0, 0, 0

    for i in range(len(y_hat)):
        if y_actual[i] == y_hat[i] == 1:
            TP += 1
        if y_hat[i] == 1 and y_actual[i] != y_hat[i]:
            FP += 1
        if y_actual[i] == y_hat[i] == 0:
            TN += 1
        if y_hat[i] == 0 and y_actual[i] != y_hat[i]:
            FN += 1

    return (TP, FP, TN, FN)


def perf_measure_by_threshold(y_actual, y_hat, threshold=0.5):
    """ Round the predictions based on the threshold first, and then count true positives,
    false positives, true negatives, and faulse negatives.

    Args:
        y_actual (array): true labels.
        y_hat (array): predictions. Floats in [0, 1].
        threshold (float): predictions smaller than the threshold value are round to 0,
            others are round to 1.

    Returns:
        TP (int): number of true positives.
        FP (int): number of false positives.
        TN (int): number of true negatives.
        FN (int): number of false negatives.
    """
    y_hat = np.where(y_hat < threshold, 0, 1)
    return perf_measure(y_actual, y_hat)
