"Machine learning functions associated to SRH parameter extraction"
from ..utils.matplotlibstyle import *
from ..utils import SaveObj, LoadObj
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import datetime
import time
from sklearn.model_selection import *
from sklearn.preprocessing import *
from sklearn.metrics import *
from sklearn.ensemble import *
from sklearn.tree import *
from sklearn.neural_network import *
from sklearn.linear_model import *
from sklearn.svm import *
from sklearn.neighbors import *
from sklearn.model_selection import *

class ML():
    #****   Constant declaration    ****#
    DefaultParameters = {}

    #****   Method declaration      ****#
    def __init__(self,dataset):
        #   Check applicability of method

        #   Create directory for computation on this dataset

        #   define hyper parameters for ML training
        self.dataset = dataset
        self.logID = None
        self.logTrain={}

    def trainRegressor(self, targetCol, trainParameters=None):
        #   define and update training parameters:
        trainParam={
            'validation_fraction': 0.01,    # validation dataset percentage
            'normalize': False,     # Wether or not to normalize the input data (True for NN)
            'base_model': RandomForestRegressor(n_estimators=100, n_jobs=-1),
            'random_seed': np.random.randint(1000),
            'bandgap': 'all', #or 'upper' == Et>0 or 'lower' ==Et<0
            'normalize': True,
            }
        if trainParameters!=None:
            for key in trainParameters.keys(): trainParam[key]=trainParameters[key]
        if trainParam['bandgap'] not in ['all','upper','lower']: raise ValueError('bandgap parameter must be all, lower or upper')
        trainKey = targetCol+"_"+trainParam['bandgap']
        self.logTrain[trainKey]={
            'target_col':targetCol,
            'train_parameters':trainParam,
        }
        #   Normalize dataset
        if trainParam['normalize']:
            scaler_dict = {}
            for col in self.dataset.columns:
                if col in ['Name','bandgap']: continue
                scaler_dict[col]=MinMaxScaler()
                self.dataset[col]=scaler_dict[col].fit_transform(self.dataset[col].values.reshape(-1,1))
        else:
            scaler_dict=None

        self.logTrain['scaler']=scaler_dict
        #   Prepare Dataset for training
        dfTrain, dfVal = train_test_split(self.dataset, test_size=trainParam['validation_fraction'],random_state=trainParam['random_seed'])
        xTrain = dfTrain.drop(["Name","Et_eV","Sn_cm2","Sp_cm2",'k','logSn','logSp','logk','bandgap'],axis =1)
        yTrain = dfTrain[targetCol]
        xVal = dfVal.drop(["Name","Et_eV","Sn_cm2","Sp_cm2",'k','logSn','logSp','logk','bandgap'],axis =1)
        yVal = dfVal[targetCol]

        #   Train model
        model = trainParam['base_model']
        training_start_time = time.time()
        model.fit(xTrain,yTrain)
        training_end_time =time.time()
        self.logTrain[trainKey]['model'] = model

        #   Score
        actTrain = yTrain
        predTrain = model.predict(xTrain)
        actVal = yVal
        predVal = model.predict(xVal)

        if trainParam['normalize']:
            actTrain = scaler_dict[targetCol].inverse_transform(actTrain.values.reshape(-1, 1))
            predTrain = scaler_dict[targetCol].inverse_transform(predTrain.reshape(-1, 1))
            actVal = scaler_dict[targetCol].inverse_transform(actVal.values.reshape(-1, 1))
            predVal = scaler_dict[targetCol].inverse_transform(predVal.reshape(-1, 1))

        self.logTrain[trainKey]['results'] = {
            "training_time": "{:.2f} s".format(training_end_time-training_start_time),
            "training_r2": "{:.3f}".format(r2_score(actTrain,predTrain)),
            "validation_r2": "{:.3f}".format(r2_score(actVal,predVal)),
            "training_rmse":"{:.2e}".format(mean_squared_error(actTrain,predTrain,squared=False)),
            "validation_rmse":"{:.2e}".format(mean_squared_error(actVal,predVal,squared=False)),
        }
        #   Save validation data
        dfVal_output = dfVal.copy(deep=True)
        if trainParam['normalize']:
            for col in dfVal.columns:
                if col in ['Name','bandgap']: continue
                dfVal_output[col]=scaler_dict[col].inverse_transform(dfVal[col].values.reshape(-1,1))
        dfVal_output['actual'] = actVal
        dfVal_output['predicted'] = predVal
        self.logTrain[trainKey]['validation_data']= dfVal_output
