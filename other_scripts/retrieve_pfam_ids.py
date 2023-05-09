import os
import json
from collections import defaultdict
from urllib.error import HTTPError
from time import sleep

import pandas as pd
from tqdm import tqdm

from microbiomemeta.data.utils import pfam_query


uid = list()
root = "./Data/"
files = ["uniprot.txt"]
for fl in files:
    with open(os.path.join(root, fl)) as f:
        for line in f:
            u = line.strip()
            uid.append(u)
full_dict = {"Protein": uid}

chem_df = pd.DataFrame(full_dict)

uniprot2pfam = defaultdict(list)
uids = set([uid for uid in chem_df["Protein"]])
attempts = 0
for uid in tqdm(uids):
    while 1:
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

with open("./Data/unprotid2pfamid.json", "w") as f:
    json.dump(uniprot2pfam, f, indent=2)
