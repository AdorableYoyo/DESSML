import os

from tqdm import tqdm

from microbiomemeta.data.utils import get_uniprotid_from_aln


def main(verbose=False):
    alignments = os.scandir(
        os.path.join("Data", "Pfam", "Pfam_aln", "alignments_hansaim")
    )
    ids = list()
    pbar = tqdm(list(alignments)) if verbose else alignments
    for aln in pbar:
        if not aln.name.endswith(".aln"):
            continue
        else:
            ids.extend(get_uniprotid_from_aln(aln.path))
    with open(os.path.join("Data", "Pfam", "Pfam_aln", "all_ids.txt"), "w") as f:
        f.write("\n".join(ids))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-v", action="store_true")
    args = parser.parse_args()
    main(args.v)
