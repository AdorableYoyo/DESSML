export CUDA_VISIBLE_DEVICES=2
exp_id=repro_chembl_to_hmdb
python -m microbiomemeta.finetuning \
    --albert_checkpoint Data/DISAE_data/albertdata/pretrained_whole_pfam/model.ckpt-1500000 \
    --frozen_list 8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23 \
    --freezing_epochs 0 \
    --gnn_type gin \
    --train_datapath Data/ChEMBL29/all_Chembl29.tsv \
    --dev_datapath Data/HMDB/Feb_13_23_dev_test/dev_47.tsv \
    --test_datapath Data/HMDB/Feb_13_23_dev_test/test_47.tsv \
    --prot2trp_path Data/Combined/proteins/triplets_in_my_data_set.pk \
    --chm2smiles_path Data/Combined/chemicals/combined_compounds.pk \
    --epoch 100 \
    --protein_embedding_type albert \
    --prot_feature_size 256 \
    --prot_max_seq_len 256 \
    --prot_dropout 0.1 \
    --lstm_embedding_size 128 \
    --lstm_num_layers 3 \
    --lstm_hidden_size 64 \
    --lstm_out_size 128 \
    --lstm_input_dropout 0.2 \
    --lstm_output_dropout 0.3 \
    --chem_dropout 0.1 \
    --chem_conv_layer_sizes 20,20,20,20 \
    --chem_feature_size 128 \
    --chem_degrees 0,1,2,3,4,5 \
    --ap_dropout 0.1 \
    --ap_feature_size 64 \
    --prediction_mode binary \
    --random_seed 705 \
    --batch 64 \
    --max_eval_steps 1000 \
    --scheduler cosineannealing \
    --lr 2e-5 \
    --l2 1e-4 \
    --num_threads 16 \
    --log info \
    > run_logs/${exp_id}.log 2>&1 &

# Notes on usage of different experiments
#Exp1 : train chembl-> test on hmdb
# --train_datapath Data/ChEMBL29/all_Chembl29.tsv \
# --dev_datapath Data/HMDB/Feb_13_23_dev_test/dev_47.tsv \
# --test_datapath Data/HMDB/Feb_13_23_dev_test/test_47.tsv \

#Exp2 : train chembl-> test on njs16
# --train_datapath Data/ChEMBL29/all_Chembl29.tsv \
# --dev_datapath Data/NJS16/Feb_2_23_dev_test/dev_27.tsv \
# --test_datapath Data/NJS16/Feb_2_23_dev_test/test_27.tsv \
# 27,37,47 are the three folds for the cross validation , please repeat 3 times and take the average as the final score

#Exp3 : train combined-> test on literature dataset
# --train_datapath Data/Combined/activities/combined_all/train_300.tsv \
# --dev_datapath Data/Combined/activities/combined_all/dev_300.tsv \
# --test_datapath Data/TestingSetFromPaper/activities_nolipids.txt \
# 100, 200, 300 are the three folds for the cross validation , please repeat 3 times and take the average as the final score

