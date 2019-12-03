#!/usr/bin/env python
# coding: utf-8

# In[202]:


import torch
import collections
import pdb
import torch.utils.data
import csv
import json
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import math
import numpy as np

def getlabel(c1,c2):
    # get log fold change of expression

    label1=math.log((float(c1)+1.0),2)
    label2=math.log((float(c2)+1.0),2)
    label=[]
    label.append(label1)
    label.append(label2)

    fold_change=(float(c2)+1.0)/(float(c1)+1.0)
    log_fold_change=math.log((fold_change),2)
    return (log_fold_change, label)

def loadDict(filename):
    # get expression value of each gene from cell*.expr.csv
    gene_dict={}
    with open(filename) as fi:
        for line in fi:
            geneID,geneExpr=line.split(',')
            gene_dict[str(geneID)]=float(geneExpr)
    fi.close()
    return(gene_dict)

gene_dict = loadDict("data/gm_5.expr.csv")

def loadData(filename,windows,gene_dict, num_hms):
    
    with open("data/gm.train.csv") as fi:
        csv_reader=csv.reader(fi)
        data = list(csv_reader)
    
        ncols=(len(data[0]))
    fi.close()

    windows = 20
    nrows=len(data)
    ngenes=nrows/windows
    nfeatures=ncols-1
    
    # testing args.n_hms vs nfeatures
    if nfeatures != num_hms:
        print("nfeatures != num_hms")
    assert(int(nfeatures) == int(num_hms))

    
    print("Number of genes: %d" % ngenes)
    print("Number of entries: %d" % nrows)
    print("Number of HMs: %d" % nfeatures)

    count = 0
    attr = collections.OrderedDict()
    alph = [chr(x) for x in range(ord('a'), ord('z') + 1)] # alphabetical key order more stable

    gene_keys = ["geneID", "expr"]
    hm_id = ["hm_" + alph[i] for i in list(range(nfeatures))]
    data_mat = np.array(data)

    for i in range(0, nrows, windows):
        geneID=str(data[i][0].split("_")[0])
    
        meta = {}
        meta["geneID"] = geneID
        meta["expr"] = gene_dict[geneID]
    
        for j in range(nfeatures):
            meta[hm_id[j]] = torch.tensor(np.array([float(z) for z in data_mat[i:(i+windows), j+1]]).reshape(windows, 1))
    
        attr[count] = meta
        count+=1    
        
    return attr

#### NEED TO EDIT THIS TO TAKE N-FEATURES

class HMData(Dataset):
    # Dataset class for loading data
    def __init__(self,data_cell1,data_cell2, n_feat, transform = None): 
        self.c1=data_cell1
        self.c2=data_cell2
        self.nfeat = n_feat
        assert (len(self.c1)==len(self.c2))
    def __len__(self):
        return len(self.c1)
    def __getitem__(self,i):
                 
        alph = [chr(x) for x in range(ord('a'), ord('z') + 1)]
        hm_id = ["hm_" + alph[y] for y in list(range(self.nfeat))]
                 
#         final_data_c1=torch.cat((self.c1[i]['hm1'],self.c1[i]['hm2'],self.c1[i]['hm3'],self.c1[i]['hm4'],self.c1[i]['hm5']),1)
#         final_data_c2=torch.cat((self.c2[i]['hm1'],self.c2[i]['hm2'],self.c2[i]['hm3'],self.c2[i]['hm4'],self.c2[i]['hm5']),1)
        
        final_data_c1 = torch.cat([self.c1[i][z] for z in hm_id], 1) 
        final_data_c2 = torch.cat([self.c2[i][w] for w in hm_id], 1)         
                
        label,orig_label=getlabel(self.c1[i]['expr'],self.c2[i]['expr'])
        b_label_c1=orig_label[0]
        b_label_c2=orig_label[1]
        assert self.c1[i]['geneID']==self.c2[i]['geneID']
        geneID=self.c1[i]['geneID']
        sample={'geneID':geneID,
               'X_A':final_data_c1,
               'X_B':final_data_c2,
               'diff':label,
               'abs_A':b_label_c1,'abs_B':b_label_c2}
        return sample
    
### NEED TO PASS args.n_hms TO HMDATA FOR nfeat ARGUMENT    
#  assert that nfeatures == args.n_hms
#  could also just pass nfeatures from loadData (still needs assert)

def load_data(args):
    '''
    Loads data into a 3D tensor for each of the 3 splits.

    '''
    print("==>loading train data")
    gene_dict1=loadDict(args.data_root+args.cell_1+".expr.csv")
    gene_dict2=loadDict(args.data_root+args.cell_2+".expr.csv")
    
    cell_train_dict1=loadData(args.data_root+"/"+args.cell_1+".train.csv",
                              args.n_bins,gene_dict1, args.n_hms) # n_hms assert    
    cell_train_dict2=loadData(args.data_root+"/"+args.cell_2+".train.csv",
                              args.n_bins,gene_dict2, args.n_hms) # n_hms assert
    train_inputs = HMData(cell_train_dict1,cell_train_dict2, args.n_hms) # added dynamic args.n_hms 
    
    print("==>loading valid data")
    cell_valid_dict1=loadData(args.data_root+"/"+args.cell_1+".valid.csv", 
                              args.n_bins,gene_dict1, args.n_hms) 
    cell_valid_dict2=loadData(args.data_root+"/"+args.cell_2+".valid.csv", 
                              args.n_bins,gene_dict2, args.n_hms)   
    valid_inputs = HMData(cell_valid_dict1,cell_valid_dict2, args.n_hms) 
    
    print("==>loading test data")
    cell_test_dict1=loadData(args.data_root+"/"+args.cell_1+".test.csv", 
                             args.n_bins,gene_dict1, args.n_hms) 
    cell_test_dict2=loadData(args.data_root+"/"+args.cell_2+".test.csv", 
                             args.n_bins,gene_dict2, args.n_hms)   
    test_inputs = HMData(cell_test_dict1,cell_test_dict2, args.n_hms)


    Train = torch.utils.data.DataLoader(train_inputs, batch_size=args.batch_size, shuffle=True)
    Valid = torch.utils.data.DataLoader(valid_inputs, batch_size=args.batch_size, shuffle=False)
    Test = torch.utils.data.DataLoader(test_inputs, batch_size=args.batch_size, shuffle=False)

    return Train,Valid,Test

