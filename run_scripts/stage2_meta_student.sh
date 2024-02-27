export CUDA_VISIBLE_DEVICES=2
split=300
blc_low=0.03
teacher_lr=2e-5
student_lr=2e-5
exp_id=repro_blc_${blc_low}_split_${split}_lr_${teacher_lr}_bs64
python -m microbiomemeta.mpl_finetune \
    --albert_checkpoint  trained_model/stage1_trained/DISAE_tr_chembl_te_hmdb_${split}.pth \
    --freezing_epochs 5 \
    --train_datapath Data/ChEMBL29/all_Chembl29.tsv \
    --dev_datapath Data/HMDB/Feb_13_23_dev_test/dev_${split}.tsv \
    --test_datapath Data/HMDB/Feb_13_23_dev_test/test_${split}.tsv \
    --ul_datapath Data/new_unlabeled_data_2023/hmdb_target_sample.tsv \
    --epoch 100 \
    --teacher_lr ${teacher_lr} \
    --student_lr ${student_lr} \
    --student_lstm_dropout 0.3 \
    --student_ap_dropout 0.1 \
    --temperature 0.7 \
    --log info \
    --blc_up  1.2 \
    --blc_low  ${blc_low} \
    --exp_id ${exp_id} \
> run_logs/${exp_id}.log 2>&1 &


# Notes on usage of different experiments
#Exp1 OOD DTI :
# --train_datapath Data/DTI/chembl_train1.tsv \
# --dev_datapath Data/DTI/chembl_dev1.tsv \
# --test_datapath Data/DTI/DTI_test_x22102.tsv \
# --ul_datapath Data/DTI/DTI_target_sam_unlabeled_x53502.tsv \
# 1,2,3 are the three folds for the cross validation , please repeat 3 times and take the average as the final score
    # --albert_checkpoint trained_model/stage1_trained/DTI_fold_1_DISAE_model_state_dict_0.pth
    # --albert_checkpoint  ... 2, 3

#Exp2 Hidden human MPI :
# --train_datapath Data/ChEMBL29/all_Chembl29.tsv \
# --dev_datapath Data/HMDB/Feb_13_23_dev_test/dev_47.tsv \
# --test_datapath Data/HMDB/Feb_13_23_dev_test/test_47.tsv \
# --ul_datapath Data/new_unlabeled_data_2023/hmdb_target_sample.tsv \
# 27,37,47 are the three folds for the cross validation , please repeat 3 times and take the average as the final score
    # --albert_checkpoint trained_model/stage1_trained/DISAE_tr_chembl_te_hmdb_47.pth
    # --albert_checkpoint  ... 37 ,27


#Exp3 :Zero-shot microbiome-human MPI:
# --train_datapath Data/Combined/activities/combined_all/train_300.tsv \
# --dev_datapath Data/Combined/activities/combined_all/dev_300.tsv \
# --test_datapath Data/TestingSetFromPaper/activities_nolipids.txt \
# --ul_datapath Data/new_unlabeled_data_2023/all_gpcr_and_other_proteins_all_chem_pairs.tsv \
# 100, 200, 300 are the three folds for the cross validation , please repeat 3 times and take the average as the final score
    # --albert_checkpoint trained_model/stage1_trained/DISAE_tr_combined_te_paper_300.pth
    # --albert_checkpoint  ... 100, 200
