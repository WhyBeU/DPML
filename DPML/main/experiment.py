"Experimental main functions"
from ..si import Cell,Defect,LTS
from ..utils.matplotlibstyle import *
import numpy as np
import os
import warnings
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


class Experiment():
    #****   Constant declaration    ****#
    DefaultParameters = {
        'name':"",  # string added to every saved file for reference
        'save': False,  # True to save a copy of the printed log, the outputed model and data
        'n_defects': 100, # Size of simulated defect data set for machine learning
        'dn_range' : np.logspace(13,17,10),# Number of points to interpolate the curves on
    }

    #****   Method declaration      ****#
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
        self.logbook = {'created': datetime.datetime.now().strftime('"%d-%m-%Y %H:%M:%S"')}
        if Parameters is not None: self.updateParameters(Parameters)
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
        plt.show()
        if plotParam['save']: plt.savefig(self.pathDic['figures']+"plotSRH"+datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")+self.parameters['name']+".png",transparent=True,bbox_inches='tight')
    def updateParameters(self,Parameters):
        for key,value in Parameters.items():
            self.parameters[key]=value
    def updateLogbook(self,logItem):
        self.logbook[logItem]=datetime.datetime.now().strftime('"%d-%m-%Y %H:%M:%S"')
