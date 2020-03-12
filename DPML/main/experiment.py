"Experimental main functions"
from ..si import Cell,Defect,LTS
from ..main import ML
from ..utils.matplotlibstyle import *
from ..utils import SaveObj, LoadObj
from ..utils import Logger
import numpy as np
import os
import warnings
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import warnings
# warnings.filterwarnings("ignore")

class Experiment():
    #****   constant declaration    ****#
    DefaultParameters = {
        'name':"",  # string added to every saved file for reference
        'save': False,  # True to save a copy of the printed log, the outputed model and data
        'logML': False, # Log copy of print to console into text file
        'n_defects': 100, # Size of simulated defect data set for machine learning
        'dn_range' : np.logspace(13,17,10),# Number of points to interpolate the curves on
        'Et_min':-0.55,  #   Minimum energy level
        'Et_max':0.55,  #   maximum energy level
        'S_min':1E-18,   #   minimum capture cross section
        'S_max':1E-12,  #   maximum capture cross section
        'Nt':1E12,  #   maximum energy level
        'noise_model':"",  #   Type of noise in SRH generation
        'noise_parameter':0, #Parameter used to vary noise level from noise model
        'check_auger':True,     #   Check if lifetime generation should be below Auger limit

    }

    #****   general methods     ****#
    def __init__(self,SaveDir,Parameters=None):
        #   Check applicability of method
        if not os.path.exists(SaveDir): raise ValueError('%s does not exists'%(SaveDir))

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

        #   define hyper parameters for experiment
        self.parameters = Experiment.DefaultParameters
        self.logbook = {'created': datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")}
        self.logDataset = None
        self.logML = None
        if Parameters is not None: self.updateParameters(Parameters)
    def loadExp(path,filename=None):
        if filename != None:
            exp = LoadObj(path,filename)
            exp.updateLogbook('Experiment_loaded_'+filename)
            return(exp)
        else:   # if no filename provided, take the latest one, if none exists, raise error.
            current_timestamp = datetime.datetime(1990, 10, 24, 16, 00, 00)
            for file in os.scandir(path):
                if not file.is_file(): continue
                if 'experimentObj' in file.name:
                    timestamp = datetime.datetime.strptime(file.name.split("_")[-1].split(".")[0],"%Y-%m-%d-%H-%M-%S")
                    if timestamp > current_timestamp:
                        filename = file.name
                        current_timestamp=timestamp
            if filename != None:
                exp = LoadObj(path,filename)
                exp.updateLogbook('Experiment_loaded_'+filename)
                return(exp)
            else:
                raise ValueError("No experimental file exists in %s"%(path))
    def saveExp(self, name=None):
        if name == None: name = self.parameters['name']
        # if self.logML !=None:
        #     for id,ML in self.logML:
        #         savePath=
        self.updateLogbook('Experiment_saved_'+name)
        SaveObj(self,self.pathDic['objects'],'experimentObj_'+name+"_"+datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))

    #****   machine learning methods     ****#
    def newML(self, datasetID=None, mlParameters=None):
        if self.logDataset == None : raise ValueError("Experiment doesn't have simulated data.")
        if datasetID==None: datasetID = str(len(self.logDataset)-1) # use latest if not defined
        if not isinstance(datasetID,str): datasetID = str(datasetID)
        mlParam = {'name':None,'save':None,'logML':None}
        if mlParameters!=None:
            for key in mlParameters.keys(): mlParam[key]=mlParameters[key]
        if mlParam['save']==None: mlParam['save']=self.parameters['save']
        if mlParam['logML']==None: mlParam['logML']=self.parameters['logML']
        if self.logML==None:
            mlID = "0"
        else:
            mlID = str(len(self.logML))
        if mlParam['name']==None:mlParam['name']= self.parameters['name']+"-#"+mlID
        mlParam['mlID']=mlID
        ml = ML(Dataset=self.logDataset[datasetID], SaveDir=self.pathDic['savedir'],Parameters=mlParam)

        #   Log change
        mlID=self.updateLogMLmodel(ml, logID=mlID)
        self.updateLogbook('ML_model_created_ID'+mlID+"_"+ml.parameters['name'])
        return(ml)
    def loadML(self, filename=None):
        if filename != None:
            ml = LoadObj(self.pathDic['objects'],filename)
        else:   # if no filename provided, take the latest one, if none exists, raise error.
            current_timestamp = datetime.datetime(1990, 10, 24, 16, 00, 00)
            for file in os.scandir(self.pathDic['objects']):
                if not file.is_file(): continue
                if 'mlObj' in file.name:
                    timestamp = datetime.datetime.strptime(file.name.split("_")[-1].split(".")[0],"%Y-%m-%d-%H-%M-%S")
                    if timestamp > current_timestamp:
                        filename = file.name
                        current_timestamp=timestamp
            if filename != None:
                ml = LoadObj(self.pathDic['objects'],filename)
            else:
                raise ValueError("No ltsDF file exists in %s"%(self.pathDic['objects']))

        if ml.parameters['mlID'] != None:
            self.updateLogMLmodel(ml,logID=ml.parameters['mlID'])
        else:
            mlID = self.updateLogMLmodel(ml)
            ml.parameters['mlID'] =mlID
        self.updateLogbook('ML_loaded_ID'+ml.parameters['mlID']+"_from_"+filename)
        if ml.logger !=None: ml.logger = Logger(ml.logger)
        return(ml)
    def predictML(self, mlIDs = None, header = None):
        #   Check for applicabiliy
        if mlIDs == None: mlIDs = [str(i) for i in range(len(self.logML))]
        if header == None: header = ['Et_eV','Sn_cm2','Sp_cm2','k','logSn','logSp','logk']
        self.predictCsv = {}
        for mlID in mlIDs:
            ml = self.logML[mlID]
            self.predictCsv[mlID]={}
            #   Create dataset feature vector
            for trainKey, mlDic in ml.logTrain.items():
                vector = [t for key in self.expKeys for t in self.expDic[key]['tau_interp']]
                if trainKey=='scaler': continue
                targetCol, bandgapParam = trainKey.rsplit('_',1)
                if mlDic['train_parameters']['normalize']:
                    #   scale the feature vector
                    if len(vector) != len(mlDic['scaler'])-len(header): raise ValueError('Feature vector is not the same size as the trained ML')
                    i=0
                    for scaler_key,scaler_value in mlDic['scaler'].items():
                        if scaler_key in header: continue
                        vector[i]=np.log10(vector[i])
                        vector[i]=scaler_value.transform(vector[i].reshape(-1,1))[0][0]
                        i+=1
                #   Call ML model and predict on sample data
                if mlDic['train_parameters']['normalize']:
                    try:
                        self.predictCsv[mlID][trainKey] = mlDic['scaler'][targetCol].inverse_transform([mlDic['model'].predict([vector])])[0][0]
                    except:
                        self.predictCsv[mlID][trainKey] = mlDic['model'].predict([vector])[0]
                else:
                    self.predictCsv[mlID][trainKey] = mlDic['model'].predict([vector])[0]

        self.updateLogbook('prediction_made')

    #****   simulation methods     ****#
    def generateDB(self):
        # Generate Random defect database
        defectDB=Defect.randomDB(
            N=self.parameters['n_defects'],
            Et_min = self.parameters['Et_min'],
            Et_max = self.parameters['Et_max'],
            S_min = self.parameters['S_min'],
            S_max = self.parameters['S_max'],
            Nt = self.parameters['Nt']
        )

        # Calculate cell level characteristic for temperature,doping
        cref = Cell(T=300,Ndop=1E15,type=self.parameters['type'])
        cellDB = [cref.changeT(T).changeNdop(Ndop) for (T,Ndop) in zip(self.parameters['temperature'],self.parameters['doping'])]

        # Calculate lifetime on database and check for Auger limit
        columns_name = ["Name","Et_eV","Sn_cm2","Sp_cm2",'k','logSn','logSp','logk','bandgap']
        ltsDB=[]
        firstPass = True
        noiseparam = 0
        for d in defectDB:
            bandgap = 1 if d.Et>0 else 0
            col = [d.name,d.Et,d.Sn,d.Sp,d.k, np.log10(d.Sn),np.log10(d.Sp),np.log10(d.k),bandgap]
            skipDefect = False
            for c in cellDB:
                if firstPass:
                    for dn in self.parameters['dn_range']: columns_name.append("%sK_%scm-3_ %scm-3" % (c.T,c.Ndop,dn))
                    skipDefect = True
                    continue
                if skipDefect: continue
                s = LTS(c,d,self.parameters['dn_range'],noise=self.parameters['noise_model'], noiseparam=self.parameters['noise_parameter'])
                if self.parameters['check_auger']:  #if break auger limit, discard defect
                    breakAuger,_ = s.checkAuger()
                    if breakAuger: skipDefect=True
                if skipDefect: continue
                for t,dn in zip(s.tauSRH_noise,s.dnrange):
                    col.append(t)
            firstPass = False
            if not skipDefect: ltsDB.append(col)
            if skipDefect:  # if we skipped a defect, add a new random one to have n_defects in the database at the end
                newDefect = Defect.randomDB(
                        N=1,
                        Et_min = self.parameters['Et_min'],
                        Et_max = self.parameters['Et_max'],
                        S_min = self.parameters['S_min'],
                        S_max = self.parameters['S_max'],
                        Nt = self.parameters['Nt']
                        )[0]
                newDefect.name = d.name
                defectDB.append(newDefect)
        ltsDF = pd.DataFrame(ltsDB)
        ltsDF.columns = columns_name
        ltsID = self.updateLogDataset(ltsDF)
        if self.parameters['save']:
            SaveObj(ltsDF,self.pathDic['objects'],'ltsDF_ID'+ltsID+"_"+datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
            self.updateLogbook('lifetime_database_saved_ID'+ltsID)

        #   Log change
        self.updateLogbook('lifetime_database_generated_ID'+ltsID)
    def interpolateSRH(self):
        #   Check applicability of method

        #   For each SRH curve, linearize and interpolate on dn_range
        for key, curve in self.expDic.items():
            c = curve['cell']
            curve['X_linear_csv']=[(1)/(c.n0+c.p0+dn) for dn in curve['dn_csv']]
            curve['X_linear_interp']=[(1)/(c.n0+c.p0+dn) for dn in self.parameters['dn_range']]
            curve['Fit_slope'],curve['Fit_intercep'] = np.polyfit(curve['X_linear_csv'],curve['tau_csv'],deg=1)
            curve['tau_interp']=[curve['Fit_slope']*X+curve['Fit_intercep'] for X in curve['X_linear_interp']]

        #   Log change
        self.updateLogbook('interpolated')
    def loadCSV(self,FilePath, Temperature, Doping, Type):
        #   Check applicability of method
        if not os.path.exists(FilePath): raise ValueError('%s does not exists'%(FilePath))
        if '.csv' not in FilePath: raise ValueError('%s is not a csv file'%(FilePath))
        if len(Temperature)!=len(Doping): raise ValueError('Doping and Temperature array needs to be the same length')
        if Type not in ['n','p']: raise ValueError('%s needs to be either n or p'%(Type))

        #   Load csv file and check the correct format
        self.pathDic['csvfile']=FilePath
        self.csvDf = pd.read_csv(FilePath)
        if len(self.csvDf.columns)!=2*len(Temperature): raise ValueError('csv file does not match to Temperature and Doping array given')

        #   Create experimental data storage for further calculations
        self.expKeys = ["Exp#"+str(i) for i in range(len(Temperature))]
        new_columns = [[k1+"_Dn",k2+"_Tau"] for k1,k2 in zip(self.expKeys,self.expKeys)]
        self.csvDf.columns = [l1 for subL in new_columns for l1 in subL]
        self.expDic={}
        for key,T,Ndop in zip(self.expKeys,Temperature,Doping):
            self.expDic[key]={
                'T':T,
                'Ndop':Ndop,
                'dn_csv':self.csvDf[key+"_Dn"].dropna().values,
                'tau_csv':self.csvDf[key+"_Tau"].dropna().values,
                'cell': Cell(T=T,Ndop=Ndop,type=Type),
            }

        #   Save parameter in Parameter dicitionary
        changedParameter = {'type':Type,'temperature':Temperature,'doping':Doping}
        self.updateParameters(changedParameter)
        #   Log change
        self.updateLogbook('csv_loaded')
    def loadLTS(self, filename=None):
        if filename != None:
            ltsDF = LoadObj(self.pathDic['objects'],filename)
            ltsID = self.updateLogDataset(ltsDF)
            self.updateLogbook('ltsDB_loaded_ID'+ltsID+"_from_"+filename)
        else:   # if no filename provided, take the latest one, if none exists, raise error.
            current_timestamp = datetime.datetime(1990, 10, 24, 16, 00, 00)
            for file in os.scandir(self.pathDic['objects']):
                if not file.is_file(): continue
                if 'ltsDF' in file.name:
                    timestamp = datetime.datetime.strptime(file.name.split("_")[-1].split(".")[0],"%Y-%m-%d-%H-%M-%S")
                    if timestamp > current_timestamp:
                        filename = file.name
                        current_timestamp=timestamp
            if filename != None:
                ltsDF = LoadObj(self.pathDic['objects'],filename)
                ltsID = self.updateLogDataset(ltsDF)
                self.updateLogbook('ltsDB_loaded_ID'+ltsID+"_from_"+filename)
            else:
                raise ValueError("No ltsDF file exists in %s"%(self.pathDic['objects']))

    #****   plotting methods     ****#
    def plotSRH(self,toPlot=None, plotParameters=None):
        #   Check applicability of method
        if toPlot==None:
            toPlot=[]
            if 'csv_loaded' in self.logbook.keys(): toPlot.append('fromCSV')
            if 'interpolated' in self.logbook.keys(): toPlot.append('fromInterpolated')

        #   define and update plot parameters:
        plotParam={
            'figsize':(8,8),
            'colorscale':plt.cm.RdYlBu_r(np.linspace(0.1,0.9,len(self.expKeys))),
            'save':self.parameters['save'],
            'xlabel':'Excess minority carrier [cm$^{-3}$]',
            'ylabel':'Lifetime [s]',
            'legend':True,
            'label_fromCSV':['Experimental - (%.0F, %.1E)'%(self.expDic[key]['T'],self.expDic[key]['Ndop']) for key in self.expKeys],
            'label_fromInterpolated':['Interpolation - (%.0F, %.1E)'%(self.expDic[key]['T'],self.expDic[key]['Ndop']) for key in self.expKeys],
            'xrange':[np.min(self.parameters['dn_range']),np.max(self.parameters['dn_range'])],
            'yrange':None,
        }
        if plotParameters!=None:
            for key in plotParameters.keys(): plotParam[key]=plotParameters[key]

        #   plot figure
        plt.figure(figsize=plotParam['figsize'])
        ax = plt.gca()
        ax.set_xlabel(plotParam['xlabel'])
        ax.set_ylabel(plotParam['ylabel'])
        k=0
        ymin,ymax=np.infty,0
        for key in self.expKeys:
            if 'fromCSV' in toPlot:
                ax.scatter(self.expDic[key]['dn_csv'],self.expDic[key]['tau_csv'],c=plotParam['colorscale'][k],label=plotParam['label_fromCSV'][k])
                if np.min(self.expDic[key]['tau_csv'])<ymin: ymin = np.min(self.expDic[key]['tau_csv'])
                if np.max(self.expDic[key]['tau_csv'])>ymax: ymax = np.max(self.expDic[key]['tau_csv'])
            if 'fromInterpolated' in toPlot:
                ax.plot( self.parameters['dn_range'],self.expDic[key]['tau_interp'],c=plotParam['colorscale'][k],label=plotParam['label_fromInterpolated'][k])
                if np.min(self.expDic[key]['tau_interp'])<ymin: ymin = np.min(self.expDic[key]['tau_interp'])
                if np.max(self.expDic[key]['tau_interp'])>ymax: ymax = np.max(self.expDic[key]['tau_interp'])
            k+=1
        if plotParam['yrange']==None:  plotParam['yrange']=[0.9*ymin,1.1*ymax]
        if plotParam['xrange']!=None: ax.set_xlim(left=plotParam['xrange'][0],right=plotParam['xrange'][1])
        if plotParam['yrange']!=None: ax.set_ylim(bottom=plotParam['yrange'][0],top=plotParam['yrange'][1])
        if plotParam['legend']: ax.legend(ncol=2,bbox_to_anchor=(1,0.5), loc='center left')
        ax.loglog()
        if plotParam['save']: plt.savefig(self.pathDic['figures']+"plotSRH"+datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")+self.parameters['name']+".png",transparent=True,bbox_inches='tight')
        plt.show()

    #****   updating methods      ****#
    def updateParameters(self,Parameters):
        for key,value in Parameters.items():
            self.parameters[key]=value
    def updateLogbook(self,logItem):
        self.logbook[logItem]=datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    def updateLogDataset(self,logItem):
        if self.logDataset==None:
            self.logDataset={"0":logItem}
            id = "0"
        else:
            found = False
            for key,value in self.logDataset.items():
                if logItem.equals(value):  id, found = key, True
            if not found:
                id = str(len(self.logDataset))
                self.logDataset[id]=logItem
        return(id)
    def updateLogMLmodel(self,logItem,logID=None):
        if self.logML==None:
            self.logML={"0":logItem}
            id = "0"
        else:
            if logID==None:
                id = str(len(self.logML))
            else:
                id = logID
            self.logML[id]=logItem
        return(id)
