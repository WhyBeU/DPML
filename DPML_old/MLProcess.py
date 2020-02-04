for i in range(1):  #   [CELL]  Imports
    import warnings
    import datetime
    from pprint import pprint
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib import ticker, cm
    from matplotlib.colors import LogNorm
    from semiconductor.recombination import SRH
    import yaml
    import os
    import sys
    import time
    import csv
    import scipy
    import seaborn as sns
    from scipy.stats import linregress
    import scipy.constants as const
    from scipy.optimize import curve_fit, minimize
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
    from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    from sklearn.linear_model import SGDClassifier
    from sklearn import preprocessing
    from sklearn.svm import SVR, SVC
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.externals import joblib
    from sklearn import metrics
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import RandomizedSearchCV, cross_validate
    from DPML.logger import Logger

class MLProcess():

    #--------   Constant definition
    WORKDIR = "C:\\Users\\z5189526\\OneDrive - UNSW\\Yoann-Projects\\1-ML-based TIDLS Solver\\04-Experimental data -ML\\ML\\"
    MAX_TAU = 0.01
    SPLIT_SIZE = 0.1
    DATAFILE = None
    NORMALIZE = False
    SCALER = False

    def __init__(self, model,name : str, save : bool):

        #--------   Define files to save
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        tracefile = MLProcess.WORKDIR+"traces\\"+timestamp+"_"+"_trace_"+name+".txt"
        logger = Logger(tracefile)

        #--------   Attributes
        self.name = name
        self.timestamp = timestamp
        self.model = model
        self.save = save
        self.tracefile = tracefile
        self.logger = logger
        self.trainNum = 0

    def loadData(datafile, normalize = False):
        MLProcess.DATAFILE = datafile
        MLProcess.NORMALIZE = normalize

        #--------   Read file, assign band-gap and log-transform capture cross section
        df = pd.read_csv(datafile,index_col=None)
        dff = df
        dff['BG'] = np.where(df.Et_eV<0,0,1)
        dff['Sn_cm2']=np.log10(df['Sn_cm2'])
        dff['Sp_cm2']=np.log10(df['Sp_cm2'])
        dff['k']=np.log10(df['k'])

        #--------   Drop rows with excessive lifetime for lack of physical meaning
        dff = dff.loc[(dff[dff.columns[5]]<MLProcess.MAX_TAU)&(dff[dff.columns[104]]<MLProcess.MAX_TAU) &(dff[dff.columns[-100]]<MLProcess.MAX_TAU)  & (dff[dff.columns[-2]]<MLProcess.MAX_TAU)]
        dff = dff.reset_index(drop=True)

        #--------   Take the log of the lifetime to reduce scale
        logdf = dff.drop(['Name','Et_eV','Sn_cm2','Sp_cm2','k','BG'], axis = 1)
        logdf = np.log10(logdf)
        logdf['Name'] = dff['Name']
        logdf['BG'] = dff['BG']
        logdf['Et_eV'] = dff['Et_eV']
        logdf['Sn_cm2'] = dff['Sn_cm2']
        logdf['Sp_cm2'] = dff['Sp_cm2']
        logdf['k'] = dff['k']
        data = logdf

        #--------   Normalize using scaler if needed
        if normalize:
            dataTransform = preprocessing.MinMaxScaler()
            EtTransform = preprocessing.MinMaxScaler()
            SnTransform = preprocessing.MinMaxScaler()
            SpTransform = preprocessing.MinMaxScaler()
            kTransform = preprocessing.MinMaxScaler()
            normdf = dff.drop(['Name','Et_eV','Sn_cm2','Sp_cm2','k','BG'], axis = 1)
            column_name = normdf.columns
            normdf = pd.DataFrame(dataTransform.fit_transform(np.log10(normdf)))
            normdf.columns = column_name
            normdf['Name'] = dff['Name']
            normdf['BG'] = dff['BG']
            normdf['Et_eV'] = EtTransform.fit_transform(dff['Et_eV'].values.reshape(-1,1))
            normdf['Sn_cm2'] = SnTransform.fit_transform(dff['Sn_cm2'].values.reshape(-1,1))
            normdf['Sp_cm2'] = SpTransform.fit_transform(dff['Sp_cm2'].values.reshape(-1,1))
            normdf['k'] = kTransform.fit_transform(dff['k'].values.reshape(-1,1))
            data = normdf
            scaler = {
                'Et_eV' : EtTransform,
                'Sn_cm2' : SnTransform,
                'Sp_cm2' : SpTransform,
                'k': kTransform,
                'data' : dataTransform,
            }
            MLProcess.SCALER = scaler

        return(data)

    def initTraining(self):

        if self.save: self.logger.open()

        #--------   Print name
        print(">"*80)
        print(" "*np.max([0,np.int((80-len(self.name))/2)])+self.name)
        print("<"*80)
        print("\n")

        #--------   Print attributes
        title = "ATTRIBUTES"
        print("="*np.max([0,np.int((40-len(title))/2)+(40-len(title))%2])+" "+title+" "+"="*np.max([0,np.int((40-len(title))/2)]))
        attr = self.__dict__
        for k in attr:
            if k == "model" : continue
            if k == "logger" : continue
            if k == "trainNum" : continue
            print("\t",k,"-"*(1+len(max(attr,key=len))-len(k)),">",attr[k])
        print("\n")
        self.regResults = []
        self.clasResults = []

        #--------   Print model
        title = "MODEL"
        print("="*np.max([0,np.int((40-len(title))/2)+(40-len(title))%2])+" "+title+" "+"="*np.max([0,np.int((40-len(title))/2)]))
        attr = self.model.get_params()
        for p in attr:
            print("\t",p,"-"*(1+len(max(attr,key=len))-len(p)),">",attr[p])
        print("\n")


        if self.save: self.logger.close()

    def prepData(self,data,predictColumn, subsetSize = None, randomSeed=None, predictType = "Regression", BG=None):

        #--------   DO NOT PROCEED IF NO DATAFILE
        if not MLProcess.DATAFILE : raise ValueError("No datafile loaded using MLProcess.loadData()")

        #--------   DO NOT PROCEED IF WRONG OR NO COLUMN TO PREDICT
        if predictColumn not in ['Et_eV','Sn_cm2','Sp_cm2','k','BG'] : raise ValueError("No or wrong column to predict, expected 'Et_eV','Sn_cm2','Sp_cm2','k','BG' ")

        #--------   DO NOT PROCEED IF WRONG BANDGAP TO PREDICT
        if BG not in [None,0,1] : raise ValueError("Wrong band-gap selection, expected None, 0 or 1 ")

        title = "DATASET PREPARATION"
        print(">"*60)
        print(" "*np.max([0,np.int((60-len(title))/2)])+title)
        print("<"*60)
        print("\n")

        #--------   Prepare data X, Y
        if not randomSeed : randomSeed = np.random.randint(100)
        localData = data
        if predictColumn == "BG": predictType = "Classification"
        predictCondition = "All"
        if BG is not None:
            predictCondition = "Lower band-gap" if BG==0 else "Upper band-gap"
            localData=localData.loc[localData['BG'] == BG]
            localData = localData.reset_index(drop=True)
        if subsetSize : localData = localData.sample(subsetSize, random_state = randomSeed)
        X = localData
        Y = localData[predictColumn]



        #--------   Save to model
        self.predictColumn = predictColumn
        self.predictType = predictType
        self.prepData_randomSeed = randomSeed
        self.subsetSize = subsetSize
        self.predictCondition = predictCondition

        if self.save: self.logger.open()

        #--------   Log Dataset information
        title = "DATASET"
        print("="*np.max([0,np.int((40-len(title))/2)+(40-len(title))%2])+" "+title+" "+"="*np.max([0,np.int((40-len(title))/2)]))
        toprint = {
            "Datafile" : MLProcess.DATAFILE,
            "Dataset normalized" : MLProcess.NORMALIZE,
            "Dataset length" : len(X),
            "Subset requested" : subsetSize,
            "Band-grap condition": predictCondition,
            "Predicted column" : predictColumn,
            "Prediction type" : predictType,
            "PrepData Random seed":randomSeed,
            }
        for k in toprint:
            print("\t",k,"-"*(1+len(max(toprint,key=len))-len(k)),">",toprint[k])
        print("\n")

        #--------   Log normalization information
        if MLProcess.NORMALIZE :
            title = "NORMALIZATION SCALER"
            print("="*np.max([0,np.int((40-len(title))/2)+(40-len(title))%2])+" "+title+" "+"="*np.max([0,np.int((40-len(title))/2)]))
            for key in MLProcess.SCALER:
                if key == 'data': continue
                print("\t MinMaxScaler : " + key)
                scaler_attr = MLProcess.SCALER[key].__dict__
                for kk in scaler_attr:
                    print("\t \t",kk,"-"*(1+len(max(scaler_attr,key=len))-len(kk)),">",scaler_attr[kk])
            print("\n")

        if self.save: self.logger.close()

        return(X,Y)

    def trainModel(self,X,Y, randomSeed = None, comment = ""):
        self.trainNum +=1
        title = "TRAINING #"+str(self.trainNum)
        print(">"*60)
        print(" "*np.max([0,np.int((60-len(title))/2)])+title)
        print("<"*60)
        print("\n")


        #--------   Train and Test set splitting
        if not randomSeed : randomSeed = np.random.randint(100)
        X_train, X_test, Y_train, Y_test= train_test_split(X,Y,test_size = MLProcess.SPLIT_SIZE,random_state=randomSeed)
        plotData = X_test[['Name','Et_eV','Sn_cm2','Sp_cm2','k','BG']]
        X_train = X_train.drop(['Name','Et_eV','Sn_cm2','Sp_cm2','k','BG'], axis = 1)
        X_test = X_test.drop(['Name','Et_eV','Sn_cm2','Sp_cm2','k','BG'], axis = 1)

        #--------   Add un-normalize data to plotData
        if MLProcess.NORMALIZE:
            for key in MLProcess.SCALER:
                if key == "data": continue
                plotData[key] = MLProcess.SCALER[key].inverse_transform(plotData[key].values.reshape(-1,1))
        print("\n")

        #--------   Prep saving files
        self.figurefile = MLProcess.WORKDIR+"figures\\"+self.timestamp+"_"+"_figure_"+self.name+"_"+str(self.trainNum)+"_"+comment+".png"
        self.predictfile = MLProcess.WORKDIR+"predicts\\"+self.timestamp+"_"+"_figure_"+self.name+"_"+str(self.trainNum)+"_"+comment+".csv"
        self.modelfile = MLProcess.WORKDIR+"models\\"+self.timestamp+"_"+"_model_"+self.name+"_"+str(self.trainNum)+"_"+comment+".sav"

        if self.save: self.logger.open()

        #--------   Print training parameters
        title = "TRAINING PARAMETERS"
        print("="*np.max([0,np.int((40-len(title))/2)+(40-len(title))%2])+" "+title+" "+"="*np.max([0,np.int((40-len(title))/2)]))
        toprint = {
            "Model file" : self.modelfile,
            "Figure file" : self.figurefile,
            "Predicted file" : self.predictfile,
            "Training ID" : self.trainNum,
            "Test/train size ratio" : MLProcess.SPLIT_SIZE,
            "Training set length" : len(X_train),
            "Testing set length" : len(X_test),
            "training Random seed" : randomSeed,
            }
        for k in toprint:
            print("\t",k,"-"*(1+len(max(toprint,key=len))-len(k)),">",toprint[k])
        print("\n")

        #--------   Print training verbose
        title = "TRAINING VERBOSE"
        print("="*np.max([0,np.int((40-len(title))/2)+(40-len(title))%2])+" "+title+" "+"="*np.max([0,np.int((40-len(title))/2)]))
        training_start_time = time.time()
        self.model.fit(X_train,Y_train)
        training_end_time = time.time()
        print("\n")
        #--------   Print Results summary amd save model
        title = "RESULTS"
        print("="*np.max([0,np.int((40-len(title))/2)+(40-len(title))%2])+" "+title+" "+"="*np.max([0,np.int((40-len(title))/2)]))
        if self.predictType == "Regression":
            results = {
                "Training title":self.name+"-"+self.predictColumn+"-Data-"+self.predictCondition+"-"+comment,
                "Training ID": str(self.trainNum),
                "Training time": "{:.2f} s".format(training_end_time-training_start_time),
                "Train set score": "{:.3f}".format(self.model.score(X_train,Y_train)),
                "Test set score": "{:.3f}".format(self.model.score(X_test,Y_test)),
                "Test set MSE":"{:.3e}".format(metrics.mean_squared_error(Y_train,self.model.predict(X_train))),
                "Test set MSE":"{:.3e}".format(metrics.mean_squared_error(Y_test,self.model.predict(X_test))),
            }
            self.regResults.append(results)
            for k in results:
                print("\t",k,"-"*(1+len(max(results,key=len))-len(k)),">",results[k])
            print("\n")

        if self.predictType == "Classification":
            results = {
                "Training title":self.name+"-"+self.predictColumn+"-Data-"+self.predictCondition+"-"+comment,
                "Training ID": str(self.trainNum),
                "Training time": "{:.2f} s".format(training_end_time-training_start_time),
                "Train set score": "{:.3f}".format(self.model.score(X_train,Y_train)),
                "Test set score": "{:.3f}".format(self.model.score(X_test,Y_test)),
                "Accuracy": "{:.3f}".format(metrics.accuracy_score(Y_test,self.model.predict(X_test))),
                "F1-score": "{:.3f}".format(metrics.f1_score(Y_test,self.model.predict(X_test))),
                "Precision": "{:.3f}".format(metrics.precision_score(Y_test,self.model.predict(X_test))),
                "Recall": "{:.3f}".format(metrics.recall_score(Y_test,self.model.predict(X_test))),
                "Area under ROC curve": "{:.3f}".format(metrics.roc_auc_score(Y_test,self.model.predict_proba(X_test)[:,0])),
            }
            self.regResults.append(results)
            for k in results:
                print("\t",k,"-"*(1+len(max(results,key=len))-len(k)),">",results[k])
            print("\n")

            title = "CLASSIFICATION REPORT"
            print("="*np.max([0,np.int((40-len(title))/2)+(40-len(title))%2])+" "+title+" "+"="*np.max([0,np.int((40-len(title))/2)]))
            print(metrics.classification_report(Y_test,self.model.predict(X_test), digits=3))

        if self.save: self.logger.close()
        if self.save: joblib.dump(self.model,self.modelfile)

        #--------   Plot graph
        plotData["Actual"] = plotData[self.predictColumn]
        if self.predictType == "Regression":
            plotData["Predict"] = MLProcess.SCALER[self.predictColumn].inverse_transform(self.model.predict(X_test).reshape(-1,1))
        if self.predictType == "Classification":
            plotData["Predict"] = self.model.predict(X_test)
            plotData["PredictProba"] = [max(p) for p in self.model.predict_proba(X_test)]

        if self.predictType == "Regression":
            for i in range(1):
                plt.close()
                plt.figure(figsize=(10,10))
                plt.title(self.name+" -#"+str(self.trainNum)+"-predict-"+self.predictColumn+"-Data-"+self.predictCondition+"-"+comment)
                ax1 = plt.gca()
                ax1.set_xlabel('Actual value')
                ax1.set_ylabel('Predicted value')
                ax1.scatter(plotData["Actual"], plotData["Predict"], c="C0", marker=".", label=self.name+" -#"+str(self.trainNum))
                #ax1.plot([min([min(plotData["Actual"]),min(plotData["Predict"])]),max([max(plotData["Actual"]),max(plotData["Predict"])])],[min([min(plotData["Actual"]),min(plotData["Predict"])]),max([max(plotData["Actual"]),max(plotData["Predict"])])],linestyle="--",c="C3", label="y=x")
                ax1.plot([MLProcess.SCALER[self.predictColumn].data_min_,MLProcess.SCALER[self.predictColumn].data_max_],[MLProcess.SCALER[self.predictColumn].data_min_,MLProcess.SCALER[self.predictColumn].data_max_], linewidth=1.25 ,linestyle="--",c="C3", label="y=x")
                ax1.legend()
                ax1.grid(which='minor',linewidth=0.5)
                ax1.grid(which='major',linewidth=0.75)
                plt.minorticks_on()
                ax1.set_axisbelow(True)
                if self.save: plt.savefig(self.figurefile,transparent=True,bbox_inches='tight')
                plt.show()
                plt.close()

        if self.predictType == "Classification":
            df_CM =pd.DataFrame(metrics.confusion_matrix(Y_test,self.model.predict(X_test)), columns=np.unique(Y))
            df_CM.index = np.unique(Y)
            cbar_ticks = [np.power(10, i) for i in range(0,2+int(np.log10(len(Y_test))))]
            for i in range(1):
                plt.close()
                plt.figure(figsize=(10,10))
                plt.title(self.name+" -#"+str(self.trainNum)+"-predict-"+self.predictColumn+"-Data-"+self.predictCondition+"-"+comment)
                ax1 = plt.gca()
                sns.heatmap(
                    df_CM,
                    annot=True,
                    ax=ax1,
                    cmap=plt.cm.viridis,
                    fmt="d",
                    square=True,
                    linewidths=.5,
                    linecolor='k',
                    vmin=0.9,
                    vmax=1+np.power(10,(1+int(np.log10(len(Y_test))))),
                    norm=mpl.colors.LogNorm(vmin=0.9,vmax=1+np.power(10,(1+int(np.log10(len(Y_test)))))),
                    cbar_kws={'ticks':cbar_ticks, 'orientation':'horizontal'},
                )
                ax1.set_xlabel('Predicted labels', fontsize=14)
                ax1.set_ylabel('Actual labels', fontsize=14)
                ax1.grid(which='minor',linewidth=0)
                ax1.grid(which='major',linewidth=0)
                plt.minorticks_on()
                if self.save: plt.savefig(self.figurefile,transparent=True,bbox_inches='tight')
                plt.show()
                plt.close()

        #--------   Save predict file
        if self.save: plotData.to_csv(self.predictfile, index = None, header=True)             #   [CELL]  MLProcess class
