import os
from collections import defaultdict
import pickle as pk

from tqdm import tqdm
import json

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
    pair_file = "ic50_pairs_multiclass.txt"
    with open(os.path.join(datapath, "chembl_uniprot_mapping.json")) as f:
        c2u_map = json.load(f)

    all_proteins = defaultdict(list)
    with open(os.path.join(datapath, pair_file), "r") as f:
        for line in f:
            chem_id, protein_id, label = line.split(",")
            all_proteins[protein_id].append((chem_id, protein_id, label))
    # -----------------
    print("Get blastp clusters...", end=" ", flush=True)
    with Blastp_result_analyzer("Data/ChEMBL29/blastp_result.txt") as analyzer:
        clusters = analyzer.get_clusters()
    print("done.")
    clusters = sorted(clusters, key=find_len(all_proteins))
    # ------------------
    train, dev, test = split_proteins(all_proteins, clusters, 10000, 10000)
    with open("Data/Combined/proteins/triplets_in_my_data_set.pk", "rb") as f:
        filter_map = pk.load(f)
    print(f"train: {len(train)}, dev: {len(dev)}, test: {len(test)}")
    # ------------------
    dir_path = os.path.join(datapath, "activity", "protein_sim_split_multiclass")
    os.makedirs(dir_path, exist_ok=True)
    for lst, s in zip([train, dev, test], ["train", "dev", "test"]):
        out_f = open(os.path.join(dir_path, f"{s}.tsv"), "w")
        for entry in lst:
            chem, prot, act = entry.split("\t")
            prot = c2u_map[prot]
            if prot in filter_map:
                out_f.write("\t".join([chem, prot, act]))
        out_f.close()


if __name__ == "__main__":
    main()
