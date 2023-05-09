#!/usr/bin/bash

GPU_ID=5
config=./microbiomemeta/albert/albert_config/albert_tiny.json
vocab=./Data/Vocab/pfam_vocab_triplets.txt
# tfrecord=( `ls ./Data/Pfam/pfam_triplet_tfrecords/*` )
# tfrecord=( `ls /raid/home/hansaimlim/Pfam/pli_bert/data/pfam_triplet_tfrecords/*` )
# tfrecord=( `ls ./Data/HMP_metagenome/HMP_corpus/triplets_tfrecords/*` )
pfam_tfrecord=( `ls /raid/home/hansaimlim/Pfam/pli_bert/data/pfam_triplet_tfrecords/*` )
hmp_tfrecord=( `ls ./Data/HMP_metagenome/HMP_corpus/triplets_tfrecords/*` )
tfrecord=( "${pfam_tfrecord[@]}" "${hmp_tfrecord[@]}" )
# output_dir=./albert_pretraining/pfam/pretrained_models/
# output_dir=./albert_pretraining/hmp_metagenome/pretrained_models/
output_dir=./albert_pretraining/pfam_hmp/pretrained_models/
learning_rate_early=0.000176 #default albert lr is 0.00176
learning_rate_mature=0.00176 #default albert lr is 0.00176
eval_steps=80000
checkpoint_steps=10000


declare -i ckpt=10000
declare -i next_ckpt=20000
declare -i increment=10000

warmup_steps=10000
warmup_iter=20 #how many times to warm up
train_iter=80 #training sessions after warm-up
CUDA_VISIBLE_DEVICES=${GPU_ID} python -m microbiomemeta.albert.run_pretraining \
    --input_file=${tfrecord} \
    --output_dir=${output_dir} \
    --albert_config_file=${config} \
    --do_train=True \
    --do_eval=True \
    --train_batch_size=256 \
    --eval_batch_size=256 \
    --max_seq_length=256 \
    --max_predictions_per_seq=40 \
    --optimizer='lamb' \
    --learning_rate=${learning_rate_early} \
    --num_train_steps=${increment} \
    --num_warmup_steps=${warmup_steps} \
    --max_eval_steps=${eval_steps} \
    --save_checkpoints_steps=${checkpoint_steps}

for i in $(seq 0 $warmup_iter)
do
	init_ckpt=${output_dir}model.ckpt-${ckpt}
	CUDA_VISIBLE_DEVICES=${GPU_ID} python -m microbiomemeta.albert.run_pretraining \
	    --input_file=${tfrecord} \
	    --output_dir=${output_dir} \
	    --albert_config_file=${config} \
	    --do_train=True \
	    --do_eval=True \
	    --train_batch_size=256 \
	    --eval_batch_size=256 \
	    --max_seq_length=256 \
	    --max_predictions_per_seq=40 \
	    --optimizer='lamb' \
	    --init_checkpoint=${init_ckpt} \
	    --learning_rate=${learning_rate_early} \
	    --num_train_steps=${next_ckpt} \
	    --num_warmup_steps=${warmup_steps} \
	    --max_eval_steps=${eval_steps} \
	    --save_checkpoints_steps=${checkpoint_steps}
	ckpt=$ckpt+$increment
	next_ckpt=$next_ckpt+$increment
done


for i in $(seq 0 $train_iter)
do
	init_ckpt=${output_dir}model.ckpt-${ckpt}
	CUDA_VISIBLE_DEVICES=${GPU_ID} python -m microbiomemeta.albert.run_pretraining \
	    --input_file=${tfrecord} \
	    --output_dir=${output_dir} \
	    --albert_config_file=${config} \
	    --do_train=True \
	    --do_eval=True \
	    --train_batch_size=256 \
	    --eval_batch_size=256 \
	    --max_seq_length=256 \
	    --max_predictions_per_seq=40 \
	    --optimizer='lamb' \
	    --learning_rate=${learning_rate_mature} \
      --init_checkpoint=${init_ckpt} \
	    --num_train_steps=${next_ckpt} \
	    --max_eval_steps=${eval_steps} \
	    --save_checkpoints_steps=${checkpoint_steps}
	ckpt=$ckpt+$increment
	next_ckpt=$next_ckpt+$increment
done

