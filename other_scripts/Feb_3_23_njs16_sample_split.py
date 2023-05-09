
'''
to use combined HMDB + CHEMBL29 as train ,
njs 16 as dev/test ood cross validation.
njs 16 is closer to the final purpose of the project , use bacteria, host to predict bacteria + human

'''
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
njs16 = pd.read_csv('/raid/home/yangliu/MicrobiomeMeta/Data/Combined/activities/njs16.tsv',sep='\t',header=None,names=['chem','host','label'])
x =  njs16[['chem','host']].values
y = njs16['label'].values
seed = 37
x1,x2,y1,y2 =train_test_split(x,y,
                    train_size=10000, test_size=10000, random_state=seed, shuffle=True, stratify=y)
dev1 = pd.DataFrame({'chem':x1[:,0],'host':x1[:,1],'label':y1})
test1 = pd.DataFrame({'chem':x2[:,0],'host':x2[:,1],'label':y2})
dev1.to_csv('/raid/home/yangliu/MicrobiomeMeta/Data/NJS16/activities/Feb_2_23_dev_test/dev_{}.tsv'.format(seed),sep='\t',header=None,index=False)
test1.to_csv('/raid/home/yangliu/MicrobiomeMeta/Data/NJS16/activities/Feb_2_23_dev_test/test_{}.tsv'.format(seed),sep='\t',header=None,index=False)