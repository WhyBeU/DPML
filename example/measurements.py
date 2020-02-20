# %%--  Playground
%reload_ext autoreload
%autoreload 2
import os
import pandas as pd

exp.__dict__

# %%-

#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#---    Initialization
#///////////////////////////////////////////
# %%--  Imports
from DPML import *
import numpy as np
# %%-


#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#---    Setup
#///////////////////////////////////////////
# %%--  Instructions:
'''
    1/ Choose SAVEDIR folder where to save the output files from DPML. (Use absolute path)
    2/ Choose FILENAME of measurements. Check out the sample.csv file for correct formattingself.
    header doesn't matter as long as they math the temperature and doping order in input
    3/ Provide TEMPERATURE as a list of the temperature in Kelvin for each measurements
    4/ Provide DOPING as a list of the temperature in cm-3 for each measurements
    5/ Provide cell type 'n' or 'p'
    6/ Change any other hyper-parameters as desired
'''
# %%-
# %%--  Inputs
SAVEDIR = "C:\\Users\\z5189526\\Documents\\GitHub\\DPML\\savedir_example\\"
FILEPATH = "C:\\Users\\z5189526\\Documents\\GitHub\\DPML\\example\\sample.csv"
TEMPERATURE = [227.3,251.8,275.8,301.4,320.5,344.3,367.9,391.3]
DOPING = [5.1e15,5.1e15,5.1e15,5.1e15,5.1e15,5.1e15,5.1e15,5.1e15]
WAFERTYPE = 'n'
# %%-
# %%--  Hyper-parameters
PARAMETERS = {
    'save': False,   # True to save a copy of the printed log, the outputed model and data
    'n_defects': 100, # Size of simulated defect data set for machine learning
    'dn_range' : np.logspace(13,17,10),# Number of points to interpolate the curves on
    'do_plot_SRH' : True, # Compute and show SRH lifetime curves
    'do_Plot_ML' : True, # Show ML training plot for the 6 regression and the classification
    'do_Plot_DPSS' : True, # Compute and show DPSS curve and ML prediction
}
# %%-

#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#---    Script
#///////////////////////////////////////////
# %%--
exp = Experiment(SaveDir=SAVEDIR, Parameters=PARAMETERS)
exp.loadCSV(FilePath=FILEPATH,Temperature=TEMPERATURE,Doping=DOPING, Type=WAFERTYPE)
exp.interpolateSRH(DnRange=Dn_Range)
exp.prepDB(N=N_Defects)
exp.trainML()   # probably need to separate classification and regression
exp.evaluate()
exp.plotSRH(DoPlot = Do_Plot_SRH)
exp.plotML(DoPlot = Do_Plot_ML)
exp.plotDPSS(DoPlot = Do_Plot_DPSS)
exp.save()
# %%-
