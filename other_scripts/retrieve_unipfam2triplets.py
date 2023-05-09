import os
import argparse
import glob

import pandas as pd
import numpy as np
import pickle5 as pickle


def parse_args():
    parser = argparse.ArgumentParser("retrieve triplets")
    parser.add_argument("--inputfile", default="covid-uni-gene-pfam.csv", type=str)
    parser.add_argument("--outputfile", default="covid", type=str)
    parser.add_argument("--mypath", default="data/ChEMBLE26/protein/", type=str)
    parser.add_argument(
        "--triplet-path",
        default="Data/Pfam/pli_bert/pfam_triplet_corpora/all_pfam_triplet_withID/",
        type=str,
        help="Path to the triples_withID file.",
    )
    opt = parser.parse_args()
    return opt


def save_dict_pickle(data, filename):
    with open(filename, "wb") as handle:
        pickle.dump(data, handle, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    opt = parse_args()
    my_path = opt.mypath
    hansaim = opt.triplet_path

    protein = pd.read_csv(my_path + opt.inputfile)
    pfam_list = list(set(list(protein.pfam.values)))
    unipfam2triplet = {}
    # missing_triplet=[]
    missing_pfam = []
    empty_pfam = []

    for pfam in pfam_list:
        print(pfam)
        care = protein[protein.pfam == pfam]
        if len(glob.glob(hansaim + pfam + "*")) > 0:
            pfam_path = glob.glob(hansaim + pfam + "*")[0]
            if os.path.exists(pfam_path):
                content = {}
                with open(pfam_path, "r") as f:
                    # i=0
                    for line in f:
                        gene = line.split("\t")[0].split("/")[0]
                        triplets = line.split("\t")[-1].split("\n")[0]
                        content[gene] = triplets
                        # i+=1
                if len(content.items()) > 0:
                    df_pfam = pd.DataFrame(content.items())
                    df_pfam.columns = ["gene", "triplets"]
                    pfam_related = care.merge(
                        df_pfam, left_on="Entry name", right_on="gene"
                    )
                    if pfam_related.shape[0] > 0:
                        key = pfam_related["uniprot|pfam"].values
                        value = pfam_related["triplets"].values
                        unipfam2triplet.update(dict(zip(key, value)))
                    # else:

                else:
                    empty_pfam.append(pfam)
        else:
            missing_pfam.append(pfam)
            print("...missing")

    save_dict_pickle(unipfam2triplet, my_path + opt.outputfile + "unipfam2triplet.pkl")
    np.save(my_path + opt.outputfile + "missing_pfam_for_triplet.npy", missing_pfam)
    np.save(my_path + opt.outputfile + "empty_pfam_for_triplet.npy", empty_pfam)
