import multiprocessing
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
import pandas as pd
from tqdm import tqdm
import random

import json
import pickle


def query_by_keys(my_dict,query_list):
    result_list = [my_dict[element] for element in query_list if element in my_dict]
    return result_list
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs

def calculate_similarity(smiles_list_1, smiles_list_2, name_list_1, name_list_2):
    # Convert SMILES strings to RDKit molecules
    mols_1 = [Chem.MolFromSmiles(smi) for smi in smiles_list_1]
    mols_2 = [Chem.MolFromSmiles(smi) for smi in smiles_list_2]

    # Calculate molecular fingerprints for each molecule
    fps_1 = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048) for mol in mols_1]
    fps_2 = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048) for mol in mols_2]

    # Calculate pairwise similarities between the fingerprints
    similarities = dict()
    for i, (fp1, name1) in tqdm(enumerate(zip(fps_1, name_list_1)), total=len(fps_1), desc='Calculating similarities'):
        for j, (fp2, name2) in enumerate(zip(fps_2, name_list_2)):
            similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
            if similarity >0.5:
                similarities[(name1, name2)] = similarity
    return similarities


if __name__ == "__main__":
    # Example usage:
    seed=72
    random.seed(seed)
    chembl = pd.read_csv("/raid/home/yoyowu/MicrobiomeMeta/Data/ChEMBL29/all_Chembl29.tsv",sep='\t',header=None, names=['chem','prot','activity'])
    hmdb_test = pd.read_csv('/raid/home/yoyowu/MicrobiomeMeta/Data/HMDB/activity/Feb_13_23_dev_test/test_47.tsv',sep='\t', names=['chem','prot','activity'])
    njs16_test_27 = pd.read_csv("/raid/home/yoyowu/MicrobiomeMeta/Data/NJS16/activities/Feb_2_23_dev_test/test_27.tsv",sep='\t',names=['chem','prot','activity'])
    combined = pd.read_csv("/raid/home/yoyowu/MicrobiomeMeta/Data/Combined/activities/combined_all/train_300.tsv",sep='\t',names=['chem','prot','activity'])
    paper = pd.read_csv("/raid/home/yoyowu/MicrobiomeMeta/Data/TestingSetFromPaper/activities_nolipids.txt",sep='\t',names=['chem','prot','activity'])
    comb_chem = list(set(combined.chem))
    paper_chem = list(set(paper.chem))

    chembl_chem = list(set(chembl.chem))

    njs16_chem = list(set(njs16_test_27.chem))
    hmdb_chem = list(set(hmdb_test.chem))
    chembl_chem = random.sample(chembl_chem,10000)

    with open("/raid/home/yoyowu/MicrobiomeMeta/Data/Combined/chemicals/combined_compounds.json") as f :
        smiles_all = json.load(f)

    chembl_smiles = query_by_keys(smiles_all,chembl_chem)
    hmdb_smiles = query_by_keys(smiles_all,hmdb_chem)
    njs16_smiles= query_by_keys(smiles_all,njs16_chem)
    combined_smiles = query_by_keys(smiles_all,comb_chem)
    paper_smiles = query_by_keys(smiles_all,paper_chem)

    chembl_chem_sim = calculate_similarity(combined_smiles,paper_smiles,comb_chem,paper_chem)


    with open('/raid/home/yoyowu/MicrobiomeMeta/Data/comb_paper_abv0.5_chem_sim.pkl', 'wb') as f:
        pickle.dump(chembl_chem_sim, f)