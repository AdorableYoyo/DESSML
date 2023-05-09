import requests as r
from Bio import SeqIO
from io import StringIO

cID='K0IND8'

baseUrl="http://www.uniprot.org/uniprot/"
currentUrl=baseUrl+cID+".fasta"
response = r.post(currentUrl)
cData=''.join(response.text)

Seq=StringIO(cData)
pSeq=list(SeqIO.parse(Seq,'fasta'))
final_seq = str(pSeq[0].seq)