# Defect Parameter extraction through Machine Learning

Package containing python source code for machine learning extraction of defect parameters from experimental files.
Follow instructions from python files in example folder on how to use the package. More information about the process can be found in the following paper: https://www.nature.com/articles/s41524-020-00410-7

Abstract - The performance of high-efficiency silicon solar cells is limited by the presence of bulk defects. Identification of these defects has the potential to improve cell performance and reliability. The impact of bulk defects on minority carrier lifetime is commonly measured using temperature- and injection-dependent lifetime spectroscopy and the defect parameters, such as its energy level and capture cross-section ratio, are usually extracted by fitting the Shockley-Read-Hall equation. We propose an alternative extraction approach by using machine learning trained on more than a million simulated lifetime curves, achieving coefficient of determinations between the true and predicted values of the defect parameters above 99%. In particular, random forest regressors, show that defect energy levels can be predicted with a high precision of ±0.02 eV, 87% of the time. The traditional approach of fitting to the Shockley-Read-Hall equation usually yields two sets of defect parameters, one in each half bandgap. The machine learning model is trained to predict the half bandgap location of the energy level, and successfully overcome the traditional approach’s limitation. The proposed approach is validated using experimental measurements, where the machine learning predicts defect energy level and capture cross-section ratio within the uncertainty range of the traditional fitting method. The successful application of machine learning in the context of bulk defect parameter extraction paves the way to more complex data-driven physical models which have the potential to overcome the limitation of traditional approaches and can be applied to other materials such as perovskite and thin film.

Figure data from the paper can be found in NPJ Comput. Mater. Data folder

Cite as : 
*Buratti, Y., Le Gia, Q. T., Dick, J., Zhu, Y. & Hameiri, Z. Extracting bulk defect parameters in silicon wafers using machine learning models. npj Computational Materials 6, 1–8 (2020)*

Related work:
*Buratti, Y., Dick, J., Gia, Q. L. & Hameiri, Z. A machine learning approach to defect parameters extraction: using random forests to inverse the Shockley-Read-Hall equation. in 46th IEEE Photovoltaic Specialist Conference 4 (2019)*


##  Requirements
Written in python 3.x
Install semiconductor package from https://github.com/MK8J
Other packages needed:
  * numpy
  * pandas
  * matplotlib
  * scipy
  * sklearn
  * pickle

##  Measurements analysis
For 1-Level one Defect analysis, opent the example/measurements.py file and edit the parameters as instructed. A saving folder is provided by default to save models, traces and figures. For further analysis or examples on how to use the package, ML-dataset-size-analysis.py and ML-main-analysis.py gives examples of the impact of the size of the simulating dataset or the machine learning model used.

### Main Steps
    1/  Choose SAVEDIR folder where to save the output files from DPML. (Use absolute path)
    2/  Choose FILEPATH of measurements. Check the sample.csv file for correct formatting.
        Each pair of columns needs to match the elements of TEMPERATURE and DOPING in order.
    3/  Provide TEMPERATURE as a list of the temperature in Kelvin for each measurements
    4/  Provide DOPING as a list of the temperature in cm-3 for each measurements
    5/  Provide cell type 'n' or 'p'
    6/  NAME your experiment

### Other notes
    Change hyper-parameters as desired.
    There are hidden parameters that can be specified in most functions, they
    use by the default the class-defined parameters
    Not executing the functions in the correct order can results in errors.
    Comment or uncomment load and save point as required.
    Note the format of lifetime measurements upload in example/sample.csv
