# DPML
Defect Parameter Extraction through Machine Learning

Need semiconductor package from https://github.com/MK8J
Other packages needed:
  numpy
  pandas
  matplotlib
  scipy
  sklearn

in your command line, run
>> pip install numpy pandas matplotlib scipy sklearn

---Main Steps---
    1/  Choose SAVEDIR folder where to save the output files from DPML. (Use absolute path)
    2/  Choose FILEPATH of measurements. Check the sample.csv file for correct formatting.
        Each pair of columns needs to match the elements of TEMPERATURE and DOPING in order.
    3/  Provide TEMPERATURE as a list of the temperature in Kelvin for each measurements
    4/  Provide DOPING as a list of the temperature in cm-3 for each measurements
    5/  Provide cell type 'n' or 'p'
    6/  NAME your experiment

---Other notes---
    Change hyper-parameters as desired.
    There are hidden parameters that can be specified in most functions, they
    use by the default the class-defined parameters
    Not executing the functions in the correct order can results in errors.
