# DESSML : Data Efficiency Semi-Supervised Meta-Learning Elucidates Understudied Interspecies Molecular Interactions
DESSML is a highly data-efficient semi-supervised meta-learning framework to investigate understudied interspecies metabolite-protein interactions (MPI). Knowledge of microbiome-human MPIs is crucial to understand the relationship between microbiome and human physiology and the evolution of an ecosystem. However, our understanding of interspecies MPIs is extremely poor due to experimental limitations. The paucity of experimental data hampers the application of machine learning to MPI predictions. DESSML successfully explores unlabeled data and transfers the information of intraspecies chemical-protein interactions to the interspecies MPI predictions. It achieves three times improvement in the prediction-recall over the baseline model. Using DESSML, we reveal novel MPIs that are validated by bioactivity assays and fill in missing links in microbiome-human interactions. DESSML is a general framework to explore previously unrecognized biological domains beyond the reach of present experimental techniques.

<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.

# Architecture

![alt text](figure1.jpg "system overview")

# Data download
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7915982.svg)](https://doi.org/10.5281/zenodo.7915982)


# **Getting Started**

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

## **Prerequisites**

To run this project, you will need to have the following installed on your local machine:

- Python 3.7 or higher
- Pytorch 1.10 or higher (with CUDA)
- Pytorch geometric
- Rdkit

## **Installing**

1. Clone this repository onto your local machine using the following command:
    
    ```
    git clone https://github.com/AdorableYoyo/DESSML
    ```
    
2. Create a new Conda environment with Python 3.9 ( 3.7 or higher) by running the following command:
    
    ```
    conda create -n DESSML python=3.9
    ```
    
3. Activate the new Conda environment by running the following command:
    
    ```
    conda activate DESSML
    ```
    
4. Install the required Python packages in the Conda environment using the following command:
    
    ```
    bash requirments.sh
    ```
    

# **Stage 1 and Stage 2 model training/evaluation : Notes on Usage of Different Experiments**

This README provides instructions on how to run both Stage 1 and Stage 2 experiments, incorporating semi-supervised training with unlabeled data. There are three experiments outlined below, along with the necessary file paths and instructions for cross-validation.

## **Experiment 1: Train on ChEMBL and Test on HMDB**

### **Stage 1:**

To run Experiment 1 in Stage 1, follow these steps:

1. Execute the command : 
    
    ```
    bash run_scripts/stage1_sup.sh
    ```
    
    with the following path :
    
    --train_datapath Data/ChEMBL29/all_Chembl29.tsv \
    --dev_datapath Data/HMDB/Feb_13_23_dev_test/dev_47.tsv \
    --test_datapath Data/HMDB/Feb_13_23_dev_test/test_47.tsv \
    
2. Repeat the above steps three times, each time using a different fold (27, 37, and 47) for cross-validation.
3. After obtaining the scores for each fold, take the average as the final score.
4. This stage will output saved model in the /experiments_logs for each run. 
5. For the purpose of reproducibility, you can skip stage1 since the trained models were saved in trained_model/

### **Stage 2:**

To run Experiment 1 in Stage 2 with semi-supervised training, follow these additional steps:

1. Execute the command 
    
    ```
    bash run_scripts/stage2_meta_student.sh
    ```
    

with the following parameters:
--ul_datapath Data/new_unlabeled_data_2023/hmdb_target_sample.tsv \
--albert_checkpoint trained_model/stage1_trained/DISAE_tr_chembl_te_hmdb_47.pth

Note: trained_model/stage1_trained/DISAE_tr_chembl_te_hmdb_47.pth is from the stage 1. It was renamed from something like :  experiment_logs/exp2023-xxxxxxxx /epoch_xx/student_state_dict.pth \ \

1. Repeat the above steps three times, using the ALBERT checkpoint corresponding to each fold (27, 37, and 47) for cross-validation.
2. Take the average score across the three repetitions as the final score.

## **Experiment 2: Train on ChEMBL and Test on NJS16**

### **Stage 1:**

To run Experiment 1 in Stage 1, follow these steps:

1. Execute the command : 
    
    ```
    bash run_scripts/stage1_sup.sh
    ```
    
    with the following path :
    
    --train_datapath Data/ChEMBL29/all_Chembl29.tsv \
    --dev_datapath Data/NJS16/Feb_2_23_dev_test/dev_27.tsv \
    --test_datapath Data/NJS16/Feb_2_23_dev_test/test_27.tsv \
    
2. Repeat the above steps three times, each time using a different fold (27, 37, and 47) for cross-validation.
3. After obtaining the scores for each fold, take the average as the final score.
4. This stage will output saved model in the /experiments_logs for each run. 
5. For the purpose of reproducibility, you can skip stage1 since the trained models were saved in trained_model/

### **Stage 2:**

To run Experiment 1 in Stage 2 with semi-supervised training, follow these additional steps:

1. Execute the command 
    
    ```
    bash run_scripts/stage2_meta_student.sh
    ```
    

with the following parameters:
--ul_datapath Data/new_unlabeled_data_2023/NJS16_target_sample.tsv
--albert_checkpoint trained_model/stage1_trained/DISAE_tr_chembl_te_njs16_27.pth

Note: trained_model/stage1_trained/DISAE_tr_chembl_te_njs16_27.pth is from the stage 1. It was renamed from something like : experiment_logs/exp2023-xxxxxxxx /epoch_xx/student_state_dict.pth \

1. Repeat the above steps three times, using the ALBERT checkpoint corresponding to each fold (27, 37, and 47) for cross-validation.
2. Take the average score across the three repetitions as the final score.

## **Experiment3: Train on the combined dataset -> Test on literature dataset**

### **Stage 1:**

To run Experiment 1 in Stage 1, follow these steps:

1. Execute the command : 
    
    ```
    bash run_scripts/stage1_sup.sh
    ```
    
    with the following path :
    
    --train_datapath  Data/Combined/activities/combined_all/train_300.tsv \
    --dev_datapath Data/Combined/activities/combined_all/dev_300.tsv \
    --test_datapath Data/TestingSetFromPaper/activities_nolipids.txt \
    
2. Repeat the above steps three times, each time using a different fold (100, 200, and 300) for cross-validation.
3. After obtaining the scores for each fold, take the average as the final score.
4. This stage will output saved model in the /experiments_logs for each run. 
5. For the purpose of reproducibility, you can skip stage1 since the trained models were saved in trained_model/

### **Stage 2:**

To run Experiment 1 in Stage 2 with semi-supervised training, follow these additional steps:

1. Execute the command 
    
    ```
    bash run_scripts/stage2_meta_student.sh
    ```
    

with the following parameters:
--ul_datapath Data/new_unlabeled_data_2023/all_gpcr_and_other_proteins_all_chem_pairs.tsv
--albert_checkpoint trained_model/stage1_trained/DISAE_tr_combined_te_paper_300.pth

1. Repeat the above steps three times, using the ALBERT checkpoint corresponding to each fold(100, 200, and 300) for cross-validation.
2. Take the average score across the three repetitions as the final score.

# Model inference

To use the trained model to predict on desired dataset, simply run : 

```
bash run_scripts/prediction.sh
```

- -state_dict_path trained_model/[xxxxx desired trained model ] 
--datapath [desired data to be predicted]
- — get_embeddings could be added to get embeddings instead of the final prediction.

## **Author**

- **You Wu, Li Xie, Yang Liu, Lei Xie**
