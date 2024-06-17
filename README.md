# Project Setup Guide

This README outlines the structure of the repository and necessary explainations to use them.

There are 3 folders in the repository 
- `EDAs` : Contains the EDA files contributed from our team members to help the whole team in understanding the dataset
- `Proposal`: Contains the proposal we submitted previously
- `Phi1.5`: The main code we used to fine tune our Phi1.5 model for this task
-   └─ `output`: Output of our model on the testing dataset

`Remark`: The files are primarly designed to be run on the google colab environment, please change the `.ipynb` for you local 

environment and paths in `train.sh` and `finetune.py` and `finetune_phi_1_5.ipynb` to your local path

Datasets are not provided as it exceed the file upload cap of github

## Prerequisites
Use the following code to install requirements
```
pip install ./Phi1.5/requirements.txt

```

