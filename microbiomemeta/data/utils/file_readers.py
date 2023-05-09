import io
from collections import defaultdict


class FastaAnalyzer:
    """ Fasta file analyzer.

    Args:
        handler (I/O instance or the content of the whole fasta file as string): the
          fasta file to be analyzed.
    """

    def __init__(self, handler):
        if isinstance(handler, str):
            handler = io.StringIO(handler)
        self.fasta = handler

    def _reset_seek_position(self):
        self.fasta.seek(0)

    def _line_loop(self, method, eof_method=None):
        self._reset_seek_position()
        while 1:
            line = self.fasta.readline()
            if isinstance(line, bytes):
                line = line.decode()
            if line == "":  # EOF
                if eof_method is not None:
                    eof_method()
                break
            else:
                method(line)

    def get_fasta_hearders(self):
        """ Read fasta file and collect all headers.

        Return:
            headers (list): Headers.
        """
        headers = list()

        def add_header(line):
            if line.startswith(">"):
                headers.append(line[1:].strip())

        self._line_loop(add_header)
        return headers

    def get_fasta_seq_length(self):
        """ Read fasta file and get the length of each sequence.

        Return:
            lengths (list): list of sequence lengths.
        """
        lens = list()
        flag = 0
        seq_len = 0

        def count_len(line):
            nonlocal flag
            nonlocal seq_len
            if line.startswith(">"):
                if flag:
                    lens.append(seq_len)
                    seq_len = 0
                else:
                    flag = 1
            else:
                seq_len += len(line.strip())

        def count_last():
            nonlocal flag
            nonlocal seq_len
            if flag:
                lens.append(seq_len)

        self._line_loop(count_len, count_last)
        return lens

    def get_fasta_seqs(self, header_handler=None):
        """ Get all fasta sequences from the fasta file.

        header_handler (callable): take in header str, output processed header as str.

        Return:
            fasta (dict): {header: fasta_sequence}.
        """
        fasta = defaultdict(str)
        cur = None

        def read_seq(line: str):
            nonlocal cur
            if line.startswith(">"):
                cur = line[1:].strip()
                if header_handler is not None:
                    cur = header_handler(cur)
            else:
                if cur is None:
                    return
                fasta[cur] += line.strip()

        self._line_loop(read_seq)
        return fasta


class ClstrAnalyzer:
    """ Analyze the cluster file output from cd-hit

    Args:
        handler (File object): the file handler of the .clstr file
    """

    def __init__(self, handler):
        self.handler = handler

    def get_clusters(self, id_sep="..."):
        """ Get the clusters in the file.

        Args:
            id_sep (str): the string used to separate the id from the cluster line. E.g.
              in case of

              "0	23217aa, >PROTEIN_A... *",

              the separator should be "...", while in case of

              "0	1364aa, >PROTEIN_A/84-1447|PROTEIN_B/82-1445|PROTEIN_C... *",

              the separator should be "|". Default is "...".

        Return (iterator):
            The method returns a iterator of the entry ids in lists.
        """
        ids = list()
        for line in self.handler:
            if line.startswith(">"):
                if len(ids) > 0:
                    yield ids
                ids = list()
            else:
                ids.append(line.split(">")[1].split(id_sep)[0])
        if len(ids) > 0:
            yield ids  # the last cluster


def read_hmmer_sto(path):
    """ Read HMMER-output Stockholm-formatted alignment file.

    Args:
        path (str): path to the alignment.

    Returns (dict):
        {gene_id: alignment line}
    """
    alignments = defaultdict(str)
    stof = open(path, "r")
    for line in stof:
        if line.startswith(("#", "//")) or len(line) <= 1:
            continue
        id, aln = [token.strip() for token in line.split()]
        alignments[id] += aln
    stof.close()
    return alignments


def get_uniprotid_from_aln(path):
    """ Get UniProtId from Pfam alignment file.

    Args:
        path (str): path to the alignment file.

    Return (list):
        list of UniProt IDs.
    """
    ids = list()
    with open(path, "r") as f:
        line = f.readline()
        while line:
            if not line.startswith("#=GS"):
                line = f.readline()
                continue
            else:
                tokens = line.split()
                if "AC" in tokens:
                    ids.append(tokens[1].split("/")[0])
            line = f.readline()
    return ids


def collect_msa(path):
    """ Read Pfam's Stockholm formatted multi-sequence alignment file.

    Args:
        path (str): path to the file

    Returns:
        msa (dict): alignment lines with sequence id as keys.
        accession (dict): {seqid: UniProt accession id}
    """
    msa = {}
    accession = {}
    with open(path, "r", encoding="utf-8", errors="replace") as inf:
        for line in inf:
            line = line.strip().split()
            if len(line) < 2:
                continue
            if line[0].startswith("#"):
                if line[0] == "#=GS":
                    if line[2] == "AC":
                        # e.g., #=GS U5QJ87_9CYAN/28-322        AC U5QJ87.1
                        seqid = line[1]
                        uni = line[3]
                        accession[seqid] = uni
                elif line[0] == "#=GC":
                    # e.g., #=GC seq_cons
                    msa["consensus"] = line[2]
            elif line[0].startswith("/"):
                break
            else:
                seqid = line[0]
                aligned_seq = line[1]
                msa[seqid] = aligned_seq
    return msa, accession
