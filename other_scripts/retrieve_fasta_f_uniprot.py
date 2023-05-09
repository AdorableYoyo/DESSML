import requests
from tqdm import tqdm
import pandas as pd
import numpy as np
import requests


njs16_test_27 = pd.read_csv("/raid/home/yoyowu/MicrobiomeMeta/Data/NJS16/activities/Feb_2_23_dev_test/test_27.tsv",sep='\t',names=['chem','prot','activity'])
#len(set(hmdb_prot).intersection(set(chembl_prot)))/len(set(hmdb_prot))
njs16_prot = list(set(njs16_test_27.prot))

# def get_protein_sequence(uniprot_id):
#     url = f"https://www.uniprot.org/uniprot/{uniprot_id}.fasta"
#     response = requests.get(url)
#     if response.ok:
#         sequence = "".join(response.text.split("\n")[1:])
#         return sequence
#     else:
#         print('not found')
#         return None

# def write_fasta_file(protein_sequences, filename):
#     with open(filename, "w") as f:
#         for uniprot_id, sequence in protein_sequences.items():
#             f.write(f">{uniprot_id}\n{sequence}\n")


def get_protein_sequence(uniprot_id):
    url = f"https://www.uniprot.org/uniprot/{uniprot_id}.fasta"
    response = requests.get(url)
    if response.ok:
        sequence = "".join(response.text.split("\n")[1:])
        return sequence
    else:
        print("not found")
        return None

def write_fasta_file(uniprot_id, sequence, filename):
    with open(filename, "a") as f:
        f.write(f">{uniprot_id}\n{sequence}\n")

if __name__ == "__main__":
    # Example usage:
    uniprot_ids = njs16_prot
    filename =  "/raid/home/yoyowu/MicrobiomeMeta/Data/NJS16/protein_sequences.fasta"
    for uniprot_id in tqdm(uniprot_ids):
        sequence = get_protein_sequence(uniprot_id)
        if sequence is not None:
            write_fasta_file(uniprot_id, sequence, filename)

# if __name__ == "__main__":
#     # Example usage:
#     uniprot_ids = njs16_prot
#     protein_sequences = {}
#     for uniprot_id in tqdm(uniprot_ids):
#         sequence = get_protein_sequence(uniprot_id)
#         if sequence is not None:
#             protein_sequences[uniprot_id] = sequence
#     write_fasta_file(protein_sequences, "/raid/home/yoyowu/MicrobiomeMeta/Data/NJS16/protein_sequences.fasta")
