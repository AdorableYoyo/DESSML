

export CUDA_VISIBLE_DEVICES=0
python -m microbiomemeta.prediction \
    --state_dict_path trained_model/amine_spec/meta_amine_spec_300.pth \
    --datapath Data/TestingSetFromPaper/activities_nolipids.txt \
    --prot2trp_path   Data/Combined/proteins/triplets_in_my_data_set.pk \
    --chm2smiles_path Data/Combined/chemicals/combined_compounds.pk \
    --prediction_mode binary  \
    --batch 32 \
    --num_threads 8 \
    --log INFO \
> run_logs/pred_on_amine_meta.log 2>&1 &
# --get_embeddings ( comment out if chemical embeddings are needed for ouput)
