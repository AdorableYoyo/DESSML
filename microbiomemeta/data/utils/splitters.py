from .processing_scripts import Blastp_result_analyzer


def _find_len(protein_dict):
    def inner(protein_list):
        length = 0
        for prot_id in protein_list:
            length += len(protein_dict[prot_id])
        return length

    return inner


def genertate_clusters(proteins, blastp_result_path):
    """
    Args:
        proteins (dict): protein dict with protein id as key and related chem_id,
            protein_id, activity label tuple as values.
        blastp_result_path (str): path to the blastp result in 0 format.
    """
    with Blastp_result_analyzer(blastp_result_path) as analyzer:
        clusters = analyzer.get_clusters()
    clusters = sorted(clusters, key=_find_len(proteins))
    return clusters


def split_proteins(all_proteins, clusters, n_dev, n_test):
    """ Split proteins into train, dev, and test sets based on clusters of similar
    proteins.

    Args:
        all_proteins (dict): dictinary mapping protein name to IDs.
        clusters (list): list of protein clusters.
        n_dev (int): number of samples needed in the developping set. The final number
          of samples in the dev set may different from this number because proteins are
          added to dev set as clusters.
        n_test (int): number of samples needed in the testing set. The final number
          of samples in the testing set may different from this number because proteins
          are added to testing set as clusters.

    Returns:
        train (set): protein IDs in the training set.
        dev (set): protein IDs in the dev set.
        test (set): protein IDs in the testing set.
    """

    def protein_list_to_entries(prot_list):
        entries = set()
        for prot in prot_list:
            entries = entries.union({"\t".join(ent) for ent in all_proteins[prot]})
        return entries

    train, dev, test = set(), set(), set()
    for cluster in clusters:
        if len(dev) < n_dev:
            dev = dev.union(protein_list_to_entries(cluster))
        elif len(test) < n_test:
            test = test.union(protein_list_to_entries(cluster))
        else:
            train = train.union(protein_list_to_entries(cluster))
    test = test - dev
    train = train - dev - test
    return train, dev, test
