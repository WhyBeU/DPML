import torch.utils.data as Data
from PIL import Image
import pandas as pd
import numpy as np


class Dataset(Data.Dataset):
    'Dataset for CNN of SRH maps'
    def __init__(self,df,Ycol,non_feature_col,dn_len,t_len, transform=None):
        self.non_feature_col=non_feature_col
        self.dn_len=dn_len
        self.t_len=t_len
        self.labels=df[Ycol].values
        self.features=df.drop(self.non_feature_col, axis=1).values
        self.transform=transform

    def row_to_img(self,row):
        'x: min Dn to max Dn -> ; y: min T to max T ||^'
        a=[row[t*self.dn_len:(t+1)*self.dn_len].tolist() for t in range(self.t_len)]
        a=np.transpose(np.log10(a))
        a_norm=(a-a.min())/(a.max()-a.min())
        img=Image.fromarray(np.uint8(a_norm*255))
        return img

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.features)

    def __getitem__(self, index):
        'Generates one sample of data'
        row=self.features[index]
        label=self.labels[index]
        img=self.row_to_img(row)
        if self.transform != None: img=self.transform(img)
        return img, label
