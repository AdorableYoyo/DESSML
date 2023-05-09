import os
from collections import defaultdict

from tqdm import tqdm

from microbiomemeta.data.utils.processing_scripts import Blastp_result_analyzer


def find_len(protein_dict):
    def inner(protein_list):
        length = 0
        for prot_id in protein_list:
            length += len(protein_dict[prot_id])
        return length

    return inner


def split_proteins(all_proteins, clusters, n_dev, n_test):
    def protein_list_to_entries(prot_list):
        entries = set()
        for prot in prot_list:
            try:
                entries = entries.union({"\t".join(ent) for ent in all_proteins[prot]})
            except KeyError:
                print(all_proteins[prot])
                raise
        return entries

    train, dev, test = set(), set(), set()
    for cluster in tqdm(clusters, desc="Splitting"):
        if len(dev) < n_dev:
            dev = dev.union(protein_list_to_entries(cluster))
        elif len(test) < n_test:
            test = test.union(protein_list_to_entries(cluster))
        else:
            train = train.union(protein_list_to_entries(cluster))
    dev = dev - test
    train = train - dev - test
    assert len(test.intersection(train)) == 0
    assert len(test.intersection(dev)) == 0
    return train, dev, test


def main():
    datapath = os.path.join("Data", "ChEMBL29")
    # ----------------
    files = ["train.tsv", "dev.tsv", "test.tsv"]
    all_proteins = defaultdict(list)
    for fl in files:
        with open(os.path.join(datapath, "activity", "chem_sim_split", fl), "r") as f:
            for line in f:
                chem_id, protein_id, label = line.split("\t")
                all_proteins[protein_id].append((chem_id, protein_id, label))
    # -----------------
    print("Get blastp clusters...", end=" ", flush=True)
    with Blastp_result_analyzer("Data/ChEMBL29/blastp_result.txt") as analyzer:
        clusters = analyzer.get_clusters()
    print("done.")
    clusters = sorted(clusters, key=find_len(all_proteins))
    # ------------------
    train, dev, test = split_proteins(all_proteins, clusters, 10000, 10000)
    print(f"train: {len(train)}, dev: {len(dev)}, test: {len(test)}")
    # ------------------
    os.makedirs(os.path.join(datapath, "activity", "protein_sim_split"), exist_ok=True)
    for lst, s in zip([train, dev, test], ["train", "dev", "test"]):
        out_f = open(
            os.path.join(datapath, "activity", "protein_sim_split", f"{s}.tsv"), "w"
        )
        for entry in lst:
            out_f.write(entry)
        out_f.close()


if __name__ == "__main__":
    main()
