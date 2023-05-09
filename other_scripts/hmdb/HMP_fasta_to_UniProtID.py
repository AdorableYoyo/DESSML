import os
from urllib.error import HTTPError
from datetime import datetime

from microbiomemeta.data.utils import GIEconvert, FastaAnalyzer


def main():
    block_size = 10000
    headers = []
    fasta_files = list(os.scandir("Data/HMP_metagenome/all_pep_20141006"))
    for fasta in fasta_files:
        if not fasta.name.endswith(".fsa"):
            continue
        else:
            with open(fasta.path, "r") as f:
                headers.extend(FastaAnalyzer(f).get_fasta_hearders())
    gene_ids = list(set([header.split()[0].strip() for header in headers]))

    n = 0
    results = ""
    attempts = 0
    now = datetime.now()
    while block_size * n < len(gene_ids):
        try:
            response = GIEconvert(
                gene_ids[block_size * n : min(block_size * (n + 1), len(gene_ids))]
            )
            results += response.split("\n", 1)[1]
            attempts = 0
        except HTTPError:
            if attempts < 5:
                attempts += 1
            else:
                attempts = 0
                n += 1
            continue
        n += 1
        if n % 10 == 0:
            print(
                f"{n*block_size}/{len(gene_ids)} entries retrieved. "
                f"Time: {(datetime.now()-now).seconds}s",
                flush=True,
            )
            now = datetime.now()
    print(
        f"{len(gene_ids)}/{len(gene_ids)} entries retrieved.", flush=True,
    )
    with open("Data/HMP_metagenome/GeneID_UniProtID.tsv", "w") as f:
        f.write(results)


if __name__ == "__main__":
    main()
