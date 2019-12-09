
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
import pandas as pd
import random

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


def loadCell1(filename,windows,gene_dict, num_hms):
    
    # chromosomes to stratify by (just keep it to the main chr)
    hg38 = pd.read_csv("data/hg38_genes_chromosome.txt", skiprows = 1, names = ("gene", "chr")) # map genes to chromosomes
    chromosomes = [str(j) for j in list(range(1, 23))] + ["X", "Y"]

    with open(filename) as fi:
        csv_reader=csv.reader(fi)
        data = list(csv_reader)
    
        ncols=(len(data[0]))
    fi.close()
    
    
    hg38 = hg38[hg38["gene"].isin(np.array(data)[:, 0])]
    hg38 = hg38[hg38["chr"].isin(chromosomes)]

    # can't cross validate if chromosome only occurs once
    for i in chromosomes:
      if int(list(hg38["chr"]).count(i)) < 2:
        hg38 = hg38[hg38["chr"] != i]

    chroms = list(np.unique(np.array(hg38["chr"])))
    random.shuffle(chroms)
    chrom_train = chroms[0:int(0.6 * len(chroms))]
    chrom_test = chroms[int(0.6 * len(chroms)):int(0.9 * len(chroms))]
    chrom_val = chroms[int(0.9 * len(chroms)):]

    temp = []
    j = 0
    for i in np.array(data)[:, 0]:
      if i in np.array(hg38["gene"]):
        temp.append(data[j])
      j += 1
    data = temp

    nrows=len(data)
    ngenes=nrows/windows
    nfeatures=ncols-1    

    train_num = 0
    test_num = 0
    val_num = 0
    attr_train = collections.OrderedDict()
    attr_test = collections.OrderedDict()
    attr_val = collections.OrderedDict()
    alph = [chr(x) for x in range(ord('a'), ord('z') + 1)] # alphabetical key order more stable

    gene_keys = ["geneID", "expr"]
    hm_id = ["hm_" + alph[i] for i in list(range(nfeatures))]
    data_mat = np.array(data)

    random.shuffle(chroms)
    chrom_train = chroms[0:int(0.6 * len(chroms))]
    chrom_test = chroms[int(0.6 * len(chroms)):int(0.9 * len(chroms))]
    chrom_val = chroms[int(0.9 * len(chroms)):]      

    for i in range(0, nrows, windows):
        geneID=str(data[i][0].split("_")[0])
    
        meta = {}
        meta["geneID"] = geneID
        meta["chr"] = hg38[hg38["gene"] == geneID].iloc[0]["chr"]
        meta["expr"] = gene_dict[geneID]
    
        for j in range(nfeatures):
            meta[hm_id[j]] = torch.tensor(np.array([float(z) for z in data_mat[i:(i+windows), j+1]]).reshape(windows, 1))
        
        if meta["chr"] in chrom_train:
          attr_train[train_num] = meta
          train_num += 1

        elif meta["chr"] in chrom_test:
          attr_test[test_num] = meta
          test_num += 1

        else:
          attr_val[val_num] = meta
          val_num += 1   
        
    return attr_train, attr_test, attr_val, chrom_train, chrom_test, chrom_val


def loadCell2(filename,windows,gene_dict, num_hms, chrom_train, chrom_test, chrom_val):
    
    # chromosomes to stratify by (just keep it to the main chr)
    hg38 = pd.read_csv("data/hg38_genes_chromosome.txt", skiprows = 1, names = ("gene", "chr")) # map genes to chromosomes
    chromosomes = [str(j) for j in list(range(1, 23))] + ["X", "Y"]

    with open(filename) as fi:
        csv_reader=csv.reader(fi)
        data = list(csv_reader)
    
        ncols=(len(data[0]))
    fi.close()
    
    
    hg38 = hg38[hg38["gene"].isin(np.array(data)[:, 0])]
    hg38 = hg38[hg38["chr"].isin(chromosomes)]

    # can't cross validate if chromosome only occurs once
    for i in chromosomes:
      if int(list(hg38["chr"]).count(i)) < 2:
        hg38 = hg38[hg38["chr"] != i]

    temp = []
    j = 0
    for i in np.array(data)[:, 0]:
      if i in np.array(hg38["gene"]):
        temp.append(data[j])
      j += 1
    data = temp

    nrows=len(data)
    ngenes=nrows/windows
    nfeatures=ncols-1    

    train_num = 0
    test_num = 0
    val_num = 0
    attr_train = collections.OrderedDict()
    attr_test = collections.OrderedDict()
    attr_val = collections.OrderedDict()
    alph = [chr(x) for x in range(ord('a'), ord('z') + 1)] # alphabetical key order more stable

    gene_keys = ["geneID", "expr"]
    hm_id = ["hm_" + alph[i] for i in list(range(nfeatures))]
    data_mat = np.array(data)
     
    for i in range(0, nrows, windows):
        geneID=str(data[i][0].split("_")[0])
    
        meta = {}
        meta["geneID"] = geneID
        meta["chr"] = hg38[hg38["gene"] == geneID].iloc[0]["chr"]
        meta["expr"] = gene_dict[geneID]
    
        for j in range(nfeatures):
            meta[hm_id[j]] = torch.tensor(np.array([float(z) for z in data_mat[i:(i+windows), j+1]]).reshape(windows, 1))
        
        if meta["chr"] in chrom_train:
          attr_train[train_num] = meta
          train_num += 1

        elif meta["chr"] in chrom_test:
          attr_test[test_num] = meta
          test_num += 1

        else:
          attr_val[val_num] = meta
          val_num += 1   
        
    return attr_train, attr_test, attr_val



def commonGenes(c1, c2):
    set1 = set(list(pd.DataFrame.from_dict(c1, orient = "index")["geneID"]))
    set2 = set(list(pd.DataFrame.from_dict(c2, orient = "index")["geneID"]))

    df1 = pd.DataFrame.from_dict(c1, orient = "index")
    df2 = pd.DataFrame.from_dict(c2, orient = "index")

    if (set1 & set2):
        common = list(set1 & set2)
    df1 = df1[df1["geneID"].isin(common)]
    df2 = df2[df2["geneID"].isin(common)]

    return df1, df2

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
      
        final_data_c1 = torch.cat([self.c1[i][z] for z in hm_id], 1) 
        final_data_c2 = torch.cat([self.c2[i][w] for w in hm_id], 1)         
                
        label,orig_label=getlabel(self.c1[i]['expr'],self.c2[i]['expr'])
        b_label_c1=orig_label[0]
        b_label_c2=orig_label[1]
        assert self.c1[i]['geneID']==self.c2[i]['geneID']
        geneID=self.c1[i]['geneID']
        chromosome = self.c1[i]["chr"]
        sample={'geneID':geneID,
                'chr':chromosome,
               'X_A':final_data_c1,
               'X_B':final_data_c2,
               'diff':label,
               'abs_A':b_label_c1,'abs_B':b_label_c2}
        return sample
    
### NEED TO PASS args.n_hms TO HMDATA FOR nfeat ARGUMENT    
#  assert that nfeatures == args.n_hms
def load_data(args):
    '''
    Loads data into a 3D tensor for processing before cross validation.

    '''
    gene_dict1=loadDict(args.data_root+"/"+args.cell_1+".expr.csv")
    gene_dict2=loadDict(args.data_root+"/"+args.cell_2+".expr.csv")
    
    c1_train, c1_test, c1_val, chr_train, chr_test, chr_val = loadCell1(args.data_root+"/"+args.cell_1+".csv",
                              args.n_bins,gene_dict1, args.n_hms) # n_hms assert 

    c2_train, c2_test, c2_val = loadCell2(args.data_root+"/"+args.cell_2+".csv",                              
                              args.n_bins,gene_dict2, args.n_hms, 
                              chr_train, chr_test, chr_val) # n_hms assert

    c1_train, c2_train = commonGenes(c1_train, c2_train)
    c1_test, c2_test = commonGenes(c1_test, c2_test)
    c1_val, c2_val = commonGenes(c1_val, c2_val)

    train = HMData(c1_train, c2_train, args.n_hms) # added dynamic args.n_hms 
    test = HMData(c1_test, c2_test, args.n_hms)
    val = HMData(c1_val, c2_val, args.n_hms)
      
    return train, test, val # return inputs to trainloading/cross validation split later

