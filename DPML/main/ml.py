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
        'mlID':None,
    }

    #****   Core methods      ****#
    def __init__(self, Dataset, SaveDir, Parameters=None):
        '''
        ---Doc---
            Description:
                Initialize ml object with passed or default parameters
            Inputs:
                Dataset     object      Dataset used for training
                SaveDir     string      Folder path to save the data if parameters['save'] is true
                Parameters  dicitionary Force or overwrite default parameters
            Outputs:
                None
        '''
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
        self.logger = None
        self.pathDic['logfile']=self.pathDic['traces']+datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")+"_"+self.parameters['name']+".txt"
        if self.parameters['logML']: self.logger = Logger(self.pathDic['logfile'])

        #   Print header for logfile
        if self.parameters['logML']: self.logger.open()
        Logger.printTitle(self.parameters['name'],titleLen=80, newLine=False)
        Logger.printTitle('HYPER-PARAMETERS',titleLen=60, newLine=False)
        Logger.printTitle('PARAMETERS',titleLen=20)
        Logger.printDic(self.parameters)
        Logger.printTitle('PATH',titleLen=20)
        Logger.printDic(self.pathDic)
        if self.parameters['logML']: self.logger.close()
    def saveML(self, name=None):
        '''
        ---Doc---
            Description:
                Save ml object with pickle
            Inputs:
                name     string      Overwrite filename
            Outputs:
                None
        '''
        if name == None: name = self.parameters['name']
        if self.logger != None: self.logger = self.logger.logfile
        SaveObj(self,self.pathDic['objects'],'mlObj_'+name+"_"+datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    def updateParameters(self,Parameters):
        '''
        ---Doc---
            Description:
                update parameters dicitionary
            Inputs:
                Parameters  dicitionary Parameters to overwrite
            Outputs:
                None
        '''
        for key,value in Parameters.items():
            self.parameters[key]=value

    #****   ML methods      ****#
    def trainRegressor(self, targetCol, trainParameters=None):
        '''
        ---Doc---
            Description:
                training pipeline for regressors. Data and results are stored on the ML object
            Inputs:
                targetCol       string      column name of the dataframe to predict
                trainParameters dicitionary training parameters to overwrite
            Outputs:
                None
        '''
        #   define and update training parameters:
        trainParam={
            'validation_fraction': 0.01,    # validation dataset percentage
            'normalize': True,     # Wether or not to normalize the input data (True for NN)
            'base_model': RandomForestRegressor(n_estimators=100, n_jobs=-1, verbose=0),
            'random_seed': np.random.randint(1000),
            'bandgap': 'all', #or 'upper' == Et>0 or 'lower' ==Et<0
            }
        if trainParameters!=None:
            for key in trainParameters.keys(): trainParam[key]=trainParameters[key]
        if trainParam['bandgap'] not in ['all','upper','lower']: raise ValueError('bandgap parameter must be all, lower or upper')
        trainKey = targetCol+"_"+trainParam['bandgap']
        self.logTrain[trainKey]={
            'target_col':targetCol,
            'prediction_type': 'regression',
            'train_parameters':trainParam,
        }
        dfAll =  self.dataset.copy(deep=True)
        for col in self.dataset.columns:
            if col not in ["Name","Et_eV","Sn_cm2","Sp_cm2",'k','logSn','logSp','logk','bandgap']: dfAll[col]=np.log10(dfAll[col])

        #   Log parameters
        if self.parameters['logML']: self.logger.open()
        Logger.printTitle('TRAINING-REG_'+trainKey,titleLen=60, newLine=False)
        Logger.printTitle('PARAMETERS',titleLen=40)
        Logger.printDic(trainParam)
        if self.parameters['logML']: self.logger.close()

        #   Normalize dataset
        if trainParam['normalize']:
            scaler_dict = {}
            for col in self.dataset.columns:
                if col in ['Name','bandgap']: continue
                scaler_dict[col]=MinMaxScaler()
                # if col not in ["Name","Et_eV","Sn_cm2","Sp_cm2",'k','logSn','logSp','logk','bandgap']: dfAll[col]=np.log10(dfAll[col])
                dfAll[col]=scaler_dict[col].fit_transform(dfAll[col].values.reshape(-1,1))
        else:
            scaler_dict=None
        self.dfAllnorm=dfAll.copy(deep=True)
        self.logTrain[trainKey]['scaler']=scaler_dict
        #   Prepare Dataset for training
        if  trainParam['bandgap'] == 'upper': dfAll = dfAll.loc[dfAll['bandgap']==1]
        if  trainParam['bandgap'] == 'lower': dfAll = dfAll.loc[dfAll['bandgap']==0]
        dfTrain, dfVal = train_test_split(dfAll, test_size=trainParam['validation_fraction'],random_state=trainParam['random_seed'])
        xTrain = dfTrain.drop(["Name","Et_eV","Sn_cm2","Sp_cm2",'k','logSn','logSp','logk','bandgap'],axis =1)
        yTrain = dfTrain[targetCol]
        xVal = dfVal.drop(["Name","Et_eV","Sn_cm2","Sp_cm2",'k','logSn','logSp','logk','bandgap'],axis =1)
        yVal = dfVal[targetCol]

        #   Train model
        if self.parameters['logML']: self.logger.open()
        Logger.printTitle('VERBOSE',titleLen=40)
        model = trainParam['base_model']
        training_start_time = time.time()
        model.fit(xTrain,yTrain)
        training_end_time =time.time()
        self.logTrain[trainKey]['model'] = model
        if self.parameters['logML']: self.logger.close()

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
            "training_rmse":"{:.2e}".format(np.sqrt(mean_squared_error(actTrain,predTrain))),
            "validation_rmse":"{:.2e}".format(np.sqrt(mean_squared_error(actVal,predVal))),
        }

        #   Log results
        if self.parameters['logML']: self.logger.open()
        Logger.printTitle('RESULTS',titleLen=40)
        Logger.printDic(self.logTrain[trainKey]['results'])
        if self.parameters['logML']: self.logger.close()

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
        '''
        ---Doc---
            Description:
                training pipeline for classifiers. Data and results are stored on the ML object
            Inputs:
                targetCol       string      column name of the dataframe to classify
                trainParameters dicitionary training parameters to overwrite
            Outputs:
                None
        '''
        #   define and update training parameters:
        trainParam={
            'validation_fraction': 0.01,    # validation dataset percentage
            'normalize': True,     # Wether or not to normalize the input data (Must be True for NN)
            'base_model': MLPClassifier((100,100),alpha=0.001, activation = 'relu', learning_rate='adaptive', verbose=0),
            'random_seed': np.random.randint(1000),
            'bandgap': 'all', #or 'upper' == Et>0 or 'lower' ==Et<0
            }
        if trainParameters!=None:
            for key in trainParameters.keys(): trainParam[key]=trainParameters[key]
        if trainParam['bandgap'] not in ['all','upper','lower']: raise ValueError('bandgap parameter must be all, lower or upper')
        trainKey = targetCol+"_"+trainParam['bandgap']
        self.logTrain[trainKey]={
            'target_col':targetCol,
            'prediction_type': 'classification',
            'train_parameters':trainParam,
        }

        dfAll =  self.dataset.copy(deep=True)
        for col in self.dataset.columns:
            if col not in ["Name","Et_eV","Sn_cm2","Sp_cm2",'k','logSn','logSp','logk','bandgap']: dfAll[col]=np.log10(dfAll[col])

        #   Normalize dataset
        if trainParam['normalize']:
            scaler_dict = {}
            for col in self.dataset.columns:
                if col in ['Name','bandgap']: continue
                scaler_dict[col]=MinMaxScaler()
                dfAll[col]=scaler_dict[col].fit_transform(dfAll[col].values.reshape(-1,1))
        else:
            scaler_dict=None

        self.logTrain[trainKey]['scaler']=scaler_dict

        #   Log parameters
        if self.parameters['logML']: self.logger.open()
        Logger.printTitle('TRAINING-CLASS_'+trainKey,titleLen=60, newLine=False)
        Logger.printTitle('PARAMETERS',titleLen=40)
        Logger.printDic(trainParam)
        if self.parameters['logML']: self.logger.close()

        #   Prepare Dataset for training
        dfTrain, dfVal = train_test_split(dfAll, test_size=trainParam['validation_fraction'],random_state=trainParam['random_seed'])
        xTrain = dfTrain.drop(["Name","Et_eV","Sn_cm2","Sp_cm2",'k','logSn','logSp','logk','bandgap'],axis =1)
        yTrain = dfTrain[targetCol]
        xVal = dfVal.drop(["Name","Et_eV","Sn_cm2","Sp_cm2",'k','logSn','logSp','logk','bandgap'],axis =1)
        yVal = dfVal[targetCol]

        #   Train model
        if self.parameters['logML']: self.logger.open()
        Logger.printTitle('VERBOSE',titleLen=40)
        model = trainParam['base_model']
        training_start_time = time.time()
        model.fit(xTrain,yTrain)
        training_end_time =time.time()
        self.logTrain[trainKey]['model'] = model
        if self.parameters['logML']: self.logger.close()

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
        CM_labels = sorted(self.dataset[targetCol].unique())
        ind = ['Pred_'+str(c) for c in CM_labels]
        col = ['Act_'+str(c) for c in CM_labels]
        self.logTrain[trainKey]['classification_report'] = classification_report(actVal,predVal, digits=3)
        self.logTrain[trainKey]['confusion_matrix'] = pd.DataFrame(confusion_matrix(actVal,predVal),columns=col,index=ind).transpose()
        #   Log results
        if self.parameters['logML']: self.logger.open()
        Logger.printTitle('RESULTS',titleLen=40)
        Logger.printDic(self.logTrain[trainKey]['results'])
        Logger.printTitle('CONFUSION MATRIX',titleLen=40)
        self.printConfusionMatrix(trainKey)
        Logger.printTitle('CLASSIFICATION REPORT',titleLen=40)
        self.printClassificationReport(trainKey)
        if self.parameters['logML']: self.logger.close()

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
        '''
        ---Doc---
            Description:
                plot true vs predicted on the validation dataset post training
            Inputs:
                trainKey        string      training results to plot
                plotParameters  dicitionary plotting parameters to overwrite
            Outputs:
                None
        '''
        if self.logTrain[trainKey]['prediction_type'] != 'regression': raise ValueError('Wrong prediction type')
        plotParam={
            'figsize':(8,8),
            'xlabel':'True value',
            'ylabel':'Predicted value',
            'show_yx':True,
            'log_plot':False,
            'scatter_alpha':0.8,
            'scatter_s':15,
            'scatter_c': 'C8',
            'save':False,
        }
        if self.parameters['save']: plotParam['save']=True
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
        if plotParam['save']:   plt.savefig(self.pathDic['figures']+datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")+"Reg_"+trainKey+".png",transparent=True,bbox_inches='tight')
        plt.show()
    def printConfusionMatrix(self,trainKey, printParameters=None):
        '''
        ---Doc---
            Description:
                function to print out confusion matrix if it exists
            Inputs:
                trainKey        string      training to print
                printParameters dicitionary print parameters to overwrite
            Outputs:
                None
        '''
        if self.logTrain[trainKey]['prediction_type'] != 'classification': raise ValueError('Wrong prediction type')
        print(self.logTrain[trainKey]['confusion_matrix'])
        print('\n')
    def printClassificationReport(self,trainKey,printParameters=None):
        '''
        ---Doc---
            Description:
                function to print out classification report if it exists
            Inputs:
                trainKey        string      training to print
                printParameters dicitionary print parameters to overwrite
            Outputs:
                None
        '''
        if self.logTrain[trainKey]['prediction_type'] != 'classification': raise ValueError('Wrong prediction type')
        print(self.logTrain[trainKey]['classification_report'])
        print('\n')

    #****   Exporting methods      ****#
