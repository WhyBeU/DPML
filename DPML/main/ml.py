"Machine learning functions associated to SRH parameter extraction"
from ..utils.matplotlibstyle import *
from ..utils import SaveObj, LoadObj
from ..utils import Logger
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
    DefaultParameters = {
        'name':'',
        'save': False,
        'logML': False,
    }

    #****   Core methods      ****#
    def __init__(self, Dataset, SaveDir, Parameters=None):

        #   Use default if not defined
        self.parameters=ML.DefaultParameters
        if Parameters is not None: self.updateParameters(Parameters)

        #   Create directory for computation on this dataset
        self.pathDic = {
            'savedir':      SaveDir,
            'figures':      SaveDir+"\\figures\\",
            'objects':      SaveDir+"\\objects\\",
            'traces':       SaveDir+"\\traces\\",
            'outputs':      SaveDir+"\\outputs\\",
        }
        for key, value in self.pathDic.items():
                if key in ['figures','objects','traces','outputs']:
                    if not os.path.exists(value):   os.makedirs(value)
        #   define hyper parameters for ML training
        self.dataset = Dataset.copy(deep=True)
        self.logTrain={}
        self.pathDic['logfile']=self.pathDic['traces']+self.parameters['name']++"_"+datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")+".txt"
        if self.parameters['logML']: self.logger = Logger(self.pathDic['logfile'])

        #   Print header for logfile
        if self.parameters['logML']: self.logger.open()
        print(">"*(Logger.TitleLength*2))
        print(" "*np.max([0,np.int(((Logger.TitleLength*2)-len(self.parameters['name']))/2)])+self.parameters['name'])
        print("<"*(Logger.TitleLength*2))
        print("\n")
        Logger.printTitle('HYPER-PARAMETERS')
        Logger.printDic(self.parameters)
        Logger.printTitle('PATH')
        Logger.printDic(self.pathDic)
        if self.parameters['logML']: self.logger.close()
    def updateParameters(self,Parameters):
        for key,value in Parameters.items():
            self.parameters[key]=value

    #****   ML methods      ****#
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
        dfAll =  self.dataset.copy(deep=True)
        #   Normalize dataset
        if trainParam['normalize']:
            scaler_dict = {}
            for col in self.dataset.columns:
                if col in ['Name','bandgap']: continue
                scaler_dict[col]=MinMaxScaler()
                if col not in ["Name","Et_eV","Sn_cm2","Sp_cm2",'k','logSn','logSp','logk','bandgap']: dfAll[col]=np.log10(dfAll[col])
                dfAll[col]=scaler_dict[col].fit_transform(dfAll[col].values.reshape(-1,1))
        else:
            scaler_dict=None

        self.logTrain['scaler']=scaler_dict
        #   Prepare Dataset for training
        if  trainParam['bandgap'] == 'upper': dfAll = dfAll.loc[dfAll['bandgap']==1]
        if  trainParam['bandgap'] == 'lower': dfAll = dfAll.loc[dfAll['bandgap']==0]
        dfTrain, dfVal = train_test_split(dfAll, test_size=trainParam['validation_fraction'],random_state=trainParam['random_seed'])
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
    def trainClassifier(self, targetCol, trainParameters=None):
        #   define and update training parameters:
        trainParam={
            'validation_fraction': 0.01,    # validation dataset percentage
            'normalize': False,     # Wether or not to normalize the input data (True for NN)
            'base_model': MLPClassifier((100,100),alpha=0.001, activation = 'relu', learning_rate='adaptive'),
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
        probaVal = model.predict_proba(xVal)

        self.logTrain[trainKey]['results'] = {
            "training_time": "{:.2f} s".format(training_end_time-training_start_time),
            "training_logloss": "{:.2e}".format(log_loss(actTrain,predTrain)),
            "validation_logloss": "{:.2e}".format(log_loss(actVal,predVal)),
            "training_accuracy":"{:.3f}".format(accuracy_score(actTrain,predTrain)),
            "validation_accuracy":"{:.3f}".format(accuracy_score(actVal,predVal)),
            "training_f1score":"{:.3f}".format(f1_score(actTrain,predTrain)),
            "validation_f1score":"{:.3f}".format(f1_score(actVal,predVal)),
            "training_precision":"{:.3f}".format(precision_score(actTrain,predTrain)),
            "validation_precision":"{:.3f}".format(precision_score(actVal,predVal)),
            "training_recall":"{:.3f}".format(recall_score(actTrain,predTrain)),
            "validation_recall":"{:.3f}".format(recall_score(actVal,predVal)),
        }

        #   Save validation data
        dfVal_output = dfVal.copy(deep=True)
        if trainParam['normalize']:
            for col in dfVal.columns:
                if col in ['Name','bandgap']: continue
                dfVal_output[col]=scaler_dict[col].inverse_transform(dfVal[col].values.reshape(-1,1))
        dfVal_output['actual'] = actVal
        dfVal_output['predicted'] = predVal
        dfVal_output['predicted_proba'] = [np.max(proba) for proba in probaVal]
        self.logTrain[trainKey]['validation_data']= dfVal_output

    #****   Plotting methods      ****#
    def plotRegressor(self,trainKey, plotParameters=None):
        plotParam={
            'figsize':(8,8),
            'xlabel':'True value',
            'ylabel':'Predicted value',
            'legend':True,
            'show_yx':True,
            'log_plot':False,
            'scatter_alpha':0.8,
            'scatter_s':15,
            'scatter_c': 'C8'
        }

        if plotParameters!=None:
            for key in plotParameters.keys(): plotParam[key]=plotParameters[key]

        targetCol, bandgapParam = trainKey.rsplit('_',1)
        df_Val = self.logTrain[trainKey]['validation_data']
        results = self.logTrain[trainKey]['results']
        plt.figure(figsize=plotParam['figsize'])
        ax = plt.gca()
        ax.set_xlabel(plotParam['xlabel'])
        ax.set_ylabel(plotParam['ylabel'])
        ax.set_aspect('equal')

        ax.annotate("$\it{R^2}=$%s"%(results['validation_r2']),xy=(0.05,0.95),xycoords='axes fraction', fontsize=14)
        ax.set_title(targetCol+ " on "+bandgapParam+" bandgap", fontsize=14)
        ax.scatter(df_Val.actual,df_Val.predicted,marker=".",alpha=plotParam['scatter_alpha'],s=plotParam['scatter_s'],c=plotParam['scatter_c'])
        if plotParam['show_yx']: ax.plot([np.min([df_Val.actual.min(),df_Val.predicted.min()]),np.max([df_Val.actual.max(),df_Val.predicted.max()])],[np.min([df_Val.actual.min(),df_Val.predicted.min()]),np.max([df_Val.actual.max(),df_Val.predicted.max()])],'k--')
