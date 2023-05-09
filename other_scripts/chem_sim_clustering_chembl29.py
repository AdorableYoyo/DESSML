import os
from collections import defaultdict
import json

import pandas as pd
from tqdm import tqdm

from slgnn.data_processing.clustering import clustering, merge_common


def get_len(cluster, smiles_dict):
    n = 0
    for smi in cluster:
        n += len(smiles_dict[smi])
    return n


def split(clusters, smiles_dict, n_dev, n_test):
    clusters = sorted(clusters, key=lambda x: len(x))
    train, dev, test = list(), list(), list()
    dev_l, test_l = 0, 0
    for cluster in clusters:
        if dev_l < n_dev:
            dev.extend(cluster)
            dev_l += get_len(cluster, smiles_dict)
        elif test_l < n_test:
            test.extend(cluster)
            test_l += get_len(cluster, smiles_dict)
        else:
            train.extend(cluster)
    return train, dev, test


def main():
    datapath = os.path.join("Data", "ChEMBL29")
    activity_df = pd.read_csv(
        os.path.join(datapath, "activities_ic50_singleProtein_valid.csv"), index_col=0
    )
    compounds = dict()
    pb = tqdm(
        zip(activity_df["Molecule ChEMBL ID"], activity_df["Smiles"]),
        desc="Generating chemical to SMILES map",
    )
    for chemid, smiles in pb:
        compounds[chemid] = smiles
    print("Loading valid protein list...", end=" ", flush=True)
    with open(os.path.join("Data", "Pfam", "pfam_triplets_map.json")) as f:
        valid_prots = set(json.load(f).keys())
    print("done.")
    with open(os.path.join("Data", "ChEMBL29", "chembl_uniprot_mapping.json")) as f:
        chem2uni = json.load(f)
    all_smiles = defaultdict(list)
    with open(os.path.join(datapath, "ic50_pairs.txt")) as f:
        for line in f:
            chem_id, protein_id, label = line.strip().split(",")
            if chem2uni[protein_id] not in valid_prots:
                continue
            all_smiles[compounds[chem_id]].append((chem_id, protein_id, label))
    clusters = clustering(list(all_smiles.keys()), verbose=True)
    print()
    merged = list(merge_common(clusters))
    train_list, dev_list, test_list = split(merged, all_smiles, 10000, 10000)
    print(f"train: {len(train_list)}, dev: {len(dev_list)}, test: {len(test_list)}")
    os.makedirs(
        os.path.join("Data", "ChEMBL29", "activity", "chem_sim_split"), exist_ok=True
    )
    for lst, s in zip([train_list, dev_list, test_list], ["train", "dev", "test"]):
        out_f = open(
            os.path.join(datapath, "activity", "chem_sim_split", f"{s}.tsv"), "w"
        )
        for smiles in lst:
            for entry in all_smiles[smiles]:
                out_f.write("\t".join(entry) + "\n")
        out_f.close()


if __name__ == "__main__":
    main()
