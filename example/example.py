# %%--  Playground
%reload_ext autoreload
%autoreload 2
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
    1/ Choose SAVEDIR folder where to save the output files from DPML.
    2/ Choose FILENAME of measurements. Check out the sample.csv file for correct formatting
    3/ Provide TEMPERATURE as a list of the temperature in Kelvin for each measurements
    4/ Provide DOPING as a list of the temperature in cm-3 for each measurements
    5/ Change any other hyper-parameters as desired
'''
# %%-
# %%--  Inputs
SAVEDIR = "SAVEDIR\\"
FILENAME = "sample.csv"
TEMPERATURE = [100,200,300]
DOPING = [1e15,1e15,1e15]
# %%-
# %%--  Hyper-parameters
Save = False    # True to save a copy of the printed log, the outputed model and data
N_Defects = 100 # Size of simulated defect data set for machine learning
Dn_Range = np.logspace(13,17,10) # Number of points to interpolate the curves on
Do_Plot_SRH = True # Compute and show SRH lifetime curves
Do_Plot_ML = True # Show ML training plot for the 6 regression and the classification
Do_Plot_DPSS = True # Compute and show DPSS curve and ML prediction
# %%-

#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#---    Script
#///////////////////////////////////////////
# %%--
exp = Experiment(SAVEDIR,FILENAME, Save=Save)
exp.loadCSV()
exp.interpolateSRH(DnRange=Dn_Range)
exp.prepDB(N=N_Defects)
exp.trainML()
exp.evaluate()
exp.plotSRH(DoPlot = Do_Plot_SRH)
exp.plotML(DoPlot = Do_Plot_ML)
exp.plotDPSS(DoPlot = Do_Plot_DPSS)
exp.save()
# %%-
