#!/bin/bash

# requires phthon=3.6
# -------------------------
conda install -c anaconda protobuf -y
conda install -c anaconda h5py=2.10.0 -y
# pip install tensorflow==2.3.0
pip install transformers[tf-cpu]==2.3.0

# windows
# pip install torch===1.7.1+cu110 torchvision===0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
# linux
# pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -y

# -------------------------
# CHEMICAL DESCRIPTOR
# ContextPred
# cuda 11.0,
# the following is to install pytorh-geometric
# pls refer to pytorch-geometric official website for installation instruction
conda install -c anaconda scipy -y
pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-1.10.0+cu113.html
pip install --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-1.10.0+cu113.html
pip install --no-index torch-cluster -f https://pytorch-geometric.com/whl/torch-1.10.0+cu113.html
pip install --no-index torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.10.0+cu113.html
# version: 1.7.2, version 2.* does not work
pip install torch-geometric
# conda install pyg -c pyg -c conda-forge -y


# -------------------------
conda install -c conda-forge rdkit -y
pip install pickle5
conda install -c anaconda pyamg -y
