import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_curve,
    RocCurveDisplay,
    precision_recall_curve,
    PrecisionRecallDisplay,
    roc_auc_score,
    average_precision_score
)
import matplotlib.pyplot as plt
#ref_preds = np.load("prediction_logs/pred06-04-2023-11-29-20_on_test/predict_logits.npy")
#print(ref_preds[0])
#ref_preds = torch.softmax(torch.Tensor(ref_preds), axis=1).numpy()
preds = np.load("prediction_logs/pred12-04-2023-12-48-34_amine_on_gpcr/predict_logits.npy")
preds = torch.softmax(torch.Tensor(preds), axis=1).numpy()
#print('the test preds size is {}'.format(ref_preds.shape[0]))
print('actual preds size is {}'.format(preds.shape[0]))
pairs_meta = pd.read_csv('Data/final_mapping/0412_gcpr_amine_pairs.txt',delimiter="\t",header=None)

scores=preds[:,1]
#labels = pd.read_csv("/raid/home/yangliu/MicrobiomeMeta/Data/TestingSetFromPaper/activities_nolipids.txt", sep="\t", header=None)[2].to_numpy()
pairs_meta=pairs_meta.rename(columns={0:'metabolites',1:'protein'})
pairs_meta['score'] = scores



opt_idx = [np.where(preds[:,1] >0.985)]
print('there are {} hits'.format(len(pairs_meta.loc[opt_idx[0]])))
pairs_meta.loc[opt_idx[0]].to_csv('/raid/home/yoyowu/MicrobiomeMeta/predictions/exp04-11-23-48_amine_epoch_60_thr_0.985_result.csv',index=False)

#pairs_meta.to_csv('/raid/home/yangliu/MicrobiomeMeta/predictions/0406_raw_result.csv',index=False)