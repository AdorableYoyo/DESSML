import os
import json
from collections import defaultdict
from urllib.error import HTTPError
from time import sleep

import pandas as pd
from tqdm import tqdm

from microbiomemeta.data.utils import pfam_query


chem_id = list()
uid = list()
act = list()
root = "./Data/ChEMBL29/activity/protein_sim_split_multiclass"
files = ["train.tsv", "dev.tsv", "test.tsv"]
for fl in files:
    with open(os.path.join(root, fl)) as f:
        for line in f:
            c, u, a = line.strip().split("\t")
            if int(a) < 3:
                chem_id.append(c)
                uid.append(u)
                act.append(a)
full_dict = {"Compound": chem_id, "Protein": uid, "Activity": act}

chem_df = pd.DataFrame(full_dict)


uniprot2pfam = defaultdict(list)
uids = set([uid for uid in chem_df["Protein"]])
attempts = 0
for uid in tqdm(uids):
    while 1:
        # try:
        #     response = pfam_query(uid)
        #     break
        # except HTTPError:
        #     sleep(1)
        # while 1:
        try:
            response = pfam_query(uid)
        except HTTPError:
            attempts += 1
            if attempts > 3:
                attempts = 0
                break
            sleep(1)
        else:
            attempts = 0
            break
    try:
        match = response["pfam"]["entry"]["matches"]["match"]
    except KeyError:
        print(response)
        continue
    if isinstance(match, list):
        for pid in match:
            uniprot2pfam[uid].append(pid["@accession"])
    else:
        uniprot2pfam[uid].append(match["@accession"])

with open("./Data/ChEMBL29/chembl29_chemblid2pfamid.json", "w") as f:
    json.dump(uniprot2pfam, f, indent=2)
