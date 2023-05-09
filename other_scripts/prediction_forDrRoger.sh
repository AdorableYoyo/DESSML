#!/bin/bash

export CUDA_VISIBLE_DEVICES=3

python -m microbiomemeta.prediction \
    --protein_embedding_type albert \
    --prot_feature_size 256 \
    --prot_max_seq_len 256 \
    --lstm_embedding_size 128 \
    --lstm_num_layers 3 \
    --lstm_hidden_size 64 \
    --lstm_out_size 128 \
    --chem_conv_layer_sizes 20,20,20,20 \
    --chem_feature_size 128 \
    --chem_degrees 0,1,2,3,4,5 \
    --state_dict_path experiment_logs/exp2022-01-23-17-21-35/epoch_88/student_state_dict.pth \
    --gnn_type gin \
    --ap_feature_size 64 \
    --datapath Data/DrRoger_radio/104775_pairs.txt \
    --prot2trp_path  Data/final_mapping/final_prots.pk \
    --chm2smiles_path Data/DrRoger_radio/compounds.json \
    --prediction_mode binary \
    --batch 64 \
    --num_threads 8 \
    --log INFO
