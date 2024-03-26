import numpy as np
import torch
import h5py
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

class read_data:

  def __init__(self,path_to_files):
    self.root_dir = path_to_files
    self.files = list(self.root_dir + file for file in os.listdir(self.root_dir))
    #self.BATCH_SIZE = batch_size
  
  def read_h5file(self, file_path, ids_flag):
    content = []
    with h5py.File(file_path, 'r') as f:
      for key in f.keys():
        dataset = f[key][()]
        content.append(int(dataset[0]) if ids_flag else dataset)
    return content

  def load_files_content (self):
    for file in self.files: 
      if 'ids' in file:
        ids = self.read_h5file(file, ids_flag=True)
      if 'vars' in file:
        vars = self.read_h5file(file, ids_flag=False)
      if 'lines' in file:
        lines = self.read_h5file(file, ids_flag=False)
      if 'labels' in file:
        labels = self.read_h5file(file, ids_flag=False)
    return ids, lines, labels, vars
  
  def __add__ (self, B):
    i1, x1, y1, v1 = self.load_files_content ()
    i2, x2, y2, v2 = B.load_files_content()

    ##  assert statement --> check that the two set have been extracted with the same 
    # window size.

    ##
    return i1+i2, x1+x2, y1+y2, v1+v2
  
  def split_data (self, combine,B,max_norm, random_seed,TEST_SIZE,VAL_SIZE, BATCH_SIZE):
    '''
    args:
      combine,B: combine set it to True if you want to combine udf and mf, B is either mf or udf
      max_norm: True if you want to normalize by the max
      random_seed: 42 most of the time
      TEST_SIZE: size of test set
      BATCH_SIZE: whatever you want
    '''
    # check for NANs
    if np.logical_and(combine, B is not None):
      ids, lines, labels, vars = self + B
    else:
      ids, lines, labels, vars = self.load_files_content()
    ## checking for nans and full 0s ###################################
    idx = []
    for i,x in enumerate(lines):
      if np.logical_and (np.sum(x[:2]) != 0, np.any(np.isnan(x))==False):
        idx.append(i)
    ####################################################################
    X, y, ids, V = np.array(lines)[idx], np.array(labels)[idx] , np.array(ids)[idx], np.array(vars)[idx]
    
    ### normalize X by max:
    if max_norm:
      for i,x in enumerate(X):
        X[i] = X[i] / np.max(np.nan_to_num(np.abs(x),0))
    #######################
        
    X_train, X_test, y_train, y_test= train_test_split (X,y, test_size= TEST_SIZE, random_state= random_seed)
    ids_train, ids_test = train_test_split (ids, test_size=TEST_SIZE, random_state=random_seed)
    vars_train, vars_test = train_test_split (V, test_size= TEST_SIZE, random_state= random_seed)

    if VAL_SIZE != 0:
      X_train, X_val, y_train, y_val = train_test_split (X_train,y_train, test_size= VAL_SIZE, random_state= random_seed)
      ids_train, ids_val = train_test_split (ids_train, test_size=VAL_SIZE, random_state=random_seed)
      X_tval = torch.tensor(X_val).type(torch.float).unsqueeze(dim=1)
      t_tval = torch.tensor(y_val)#.type(torch.float).unsqueeze(dim=1)
      val_data  = torch.utils.data.TensorDataset(X_tval, t_tval)
      val_dataloader  = DataLoader(val_data,  BATCH_SIZE, shuffle=False)

    X_ttrain = torch.tensor(X_train).type(torch.float).unsqueeze(dim=1) # float --> 32 bits by default
    t_ttrain = torch.tensor(y_train)#.type(torch.float).unsqueeze(dim=1) # int

    X_ttest = torch.tensor(X_test).type(torch.float).unsqueeze(dim=1)
    t_ttest = torch.tensor(y_test)#.type(torch.float).unsqueeze(dim=1)

    train_data = torch.utils.data.TensorDataset(X_ttrain, t_ttrain)
    test_data  = torch.utils.data.TensorDataset(X_ttest, t_ttest)

    torch.manual_seed(42)
    train_dataloader = DataLoader(train_data, BATCH_SIZE, shuffle=True)
    test_dataloader  = DataLoader(test_data,  BATCH_SIZE, shuffle=False)

    #print(f"Dataloaders: {train_dataloader, test_dataloader}")
    print('****')
    print(f"Length of train dataloader: {len(train_dataloader)} batches of {BATCH_SIZE}")
    print(f"Length of test dataloader: {len(test_dataloader)} batches of {BATCH_SIZE}")
    if VAL_SIZE != 0: print(f"Length of val dataloader: {len(val_dataloader)} batches of {BATCH_SIZE}") 
    ## add a branch to return validation set, ids and vars if needed 
    if VAL_SIZE != 0:
      return ids_train, train_dataloader, ids_test, test_dataloader, ids_val, val_dataloader
    else:
      return ids_train, train_dataloader,ids_test, test_dataloader


def plot_confusion_matrix (net,dataloader, save_path):
    '''
    plot confusion matrix
    '''
    data, truths, preds = [], [], []
    net.eval()
    with torch.inference_mode():
      for X, y in tqdm(dataloader, desc="Making predictions"):
        y_logit = net(X)
        y_pred = torch.softmax(y_logit, dim=1).argmax(dim=1) 
        preds.append(y_pred.cpu())
        truths.append(y)
        data.append(X)

    preds = torch.cat(preds)
    truths = torch.cat(truths)
    data = torch.cat(data)
    CM = confusion_matrix(truths, preds)

    disp = ConfusionMatrixDisplay(confusion_matrix=CM, display_labels=['Class 0', 'Class 1', 'Class 2'])
    plt.figure(figsize=(3, 3))
    disp.plot()
    plt.title('Confusion Matrix')
    plt.show()
    plt.savefig(save_path, format='png')

    return data, truths, preds