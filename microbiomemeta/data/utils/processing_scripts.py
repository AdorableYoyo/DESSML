import os

from .conservation import select_positions_by_subcluster_consensus


def preprocess_hmp_metagenome(
    cluster, out_dir, matrix, max_seq_len=253, padchar="9", gapchar="j", verbose=False
):
    """ Extract singlets and triplets from HMP metagenome dataset.

    Args:
        cluster (Tuple(str, dict)): (cluster_id, {sequence_id: alignment string})
        out_dir (str): diretory to save the outputs.
        matrix (dict): matrix used to compute conservation score.
        padchar (chr): the character used to replace "." and insertions.
        gapchar (chr): the character used to replace "-".
        max_seq_len (int): the length of the representative sentences.
        verbose (bool): print complete info.

    Returns (None)
    """
    cluster_id, alns = cluster

    seqlist = [aln for aln in alns.values()]
    selected_positions = select_positions_by_subcluster_consensus(
        seqlist, matrix, max_seq_len, ncpu=1
    )

    if len(selected_positions) == 0:
        return

    singlets_dir = os.path.join(out_dir, "singlets")
    triplets_dir = os.path.join(out_dir, "triplets")
    sin_repr_dir = os.path.join(out_dir, "singlet_representatives")
    tri_repr_dir = os.path.join(out_dir, "triplet_representatives")
    for dir in [singlets_dir, triplets_dir, sin_repr_dir, tri_repr_dir]:
        os.makedirs(dir, exist_ok=True)

    all_singlets = open(
        os.path.join(singlets_dir, f"hmp_{cluster_id}_singlets.txt"), "w"
    )
    all_triplets = open(
        os.path.join(triplets_dir, f"hmp_{cluster_id}_triplets.txt"), "w"
    )
    clustered_singlets = open(
        os.path.join(sin_repr_dir, f"hmp_{cluster_id}_represent_singlets.txt"), "w"
    )
    clustered_triplets = open(
        os.path.join(tri_repr_dir, f"hmp_{cluster_id}_represent_triplets.txt"), "w"
    )

    for i_aln, seqid in enumerate(alns.keys()):
        singlets = []
        triplets = []
        for pos in selected_positions:
            if pos == 0:
                first = padchar
            else:
                first = alns[seqid][pos - 1]
            second = alns[seqid][pos]
            if (pos + 1) == len(alns[seqid]):
                third = padchar
            else:
                third = alns[seqid][pos + 1]
            singlets.append(
                alns[seqid][pos].replace(".", padchar).replace("-", gapchar).lower()
            )

            triplet = (
                "".join((first, second, third))
                .lower()
                .replace(".", padchar)
                .replace("-", gapchar)
            )
            triplets.append(triplet)
        if len(set(singlets).difference({"9", "j", "x"})) == 0:
            # skip sequences having only gaps or unknowns
            continue

        sentence_singlet = " ".join(singlets)
        sentence_triplet = " ".join(triplets)

        if i_aln == 0:  # representative sequence for the cluster
            # it is used for pretraining. No sequence ID in the file needed
            clustered_singlets.write(seqid + "\t" + sentence_singlet + "\n")
            clustered_triplets.write(seqid + "\t" + sentence_triplet + "\n")
        all_singlets.write(seqid + "\t" + sentence_singlet + "\n")
        all_triplets.write(seqid + "\t" + sentence_triplet + "\n")

    all_singlets.close()
    all_triplets.close()
    clustered_singlets.close()
    clustered_triplets.close()

    if verbose:
        print(f"{cluster_id} finished.", flush=True)


class Blastp_result_analyzer:
    """ Analyze the blastp output.

    Args:
        path (str): path to the blastp output file.
    """

    def __init__(self, path):
        self.path = path

    def __enter__(self):
        self.fh = open(self.path, "r")
        return self

    def __exit__(self, type, value, traceback):
        self.fh.close()

    # def get_pairs(self):
    #     pairs = defaultdict(list)
    #     line = self.fh.readline()
    #     while line:
    #         if line.startswith("Query="):
    #             print("\r" + line.strip(), end="")
    #             query = line.split("=")[1].strip()
    #             line = self.fh.readline()
    #             while line and not line.startswith("Query="):
    #                 if line.startswith(">"):
    #                     sbjct = line[1:].strip()
    #                     if query != sbjct:
    #                         pairs[query].append(sbjct)
    #                 line = self.fh.readline()
    #     return pairs

    def get_clusters(self):
        """ Get clusters in the blastp output file.

        Return:
            clusters (list): list of clusters. Each cluster is a set of protein IDs.
        """
        clusters = list()
        for block in self._get_blocks():
            clusters.append(self._analyze_block(block))
        return clusters

    def _get_blocks(self):
        line = self.fh.readline()
        while line:
            while line and not line.startswith("Query="):
                line = self.fh.readline()
            block = ""
            while line and not line.startswith(">"):
                block += line
                line = self.fh.readline()
            if len(block) > 0:
                yield block

    def _analyze_block(self, block):
        cluster = set()
        lines = block.split("\n")
        cluster.add(lines[0].split("=")[1].split()[0].strip())
        flag = 0
        for line in lines:
            if line.startswith("Sequences producing significant alignments"):
                flag = 1
                continue
            if flag:
                if len(line) > 1:
                    sbjct = line.split()[0]
                    cluster.add(sbjct)
        return cluster
