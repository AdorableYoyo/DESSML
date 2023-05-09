# install blast with conda:
# conda install -c bioconda blast

# ----------- for hmdb dataset
# makeblastdb -in my.fasta -dbtype prot
# blastp -db my.fasta -query my.fasta -num_threads 8 -outfmt 0 -out results.txt

# ---------- for chembl29 dataset
# makeblastdb -in Data/ChEMBL29/chembl_29.fa -dbtype prot
# blastp -db Data/ChEMBL29/chembl_29.fa -query Data/ChEMBL29/chembl_29.fa -num_threads 8 -outfmt 0 -out Data/ChEMBL29/blastp_result.txt

makeblastdb -in /raid/home/yoyowu/MicrobiomeMeta/Data/ChEMBL29/train_chembl_29.fa -dbtype prot
blastp -db /raid/home/yoyowu/MicrobiomeMeta/Data/ChEMBL29/train_chembl_29.fa -query Data/NJS16/protein_sequences.fasta -evalue 100 -num_threads 8 -outfmt 6 -out Data/ChEMBL_njs16_blast_e_100res.csv

