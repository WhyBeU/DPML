# Defect Parameter extraction through Machine Learning

Package containing python source code for machine learning extraction of defect parameters from experimental files.
Follow instructions from python files in example folder on how to use the package. More information about the process can be found in the following paper: https://www.nature.com/articles/s41524-020-00410-7

Abstract - The performance of high-efficiency silicon solar cells is limited by the presence of bulk defects. Identification of these defects has the potential to improve cell performance and reliability. The impact of bulk defects on minority carrier lifetime is commonly measured using temperature- and injection-dependent lifetime spectroscopy and the defect parameters, such as its energy level and capture cross-section ratio, are usually extracted by fitting the Shockley-Read-Hall equation. We propose an alternative extraction approach by using machine learning trained on more than a million simulated lifetime curves, achieving coefficient of determinations between the true and predicted values of the defect parameters above 99%. In particular, random forest regressors, show that defect energy levels can be predicted with a high precision of ±0.02 eV, 87% of the time. The traditional approach of fitting to the Shockley-Read-Hall equation usually yields two sets of defect parameters, one in each half bandgap. The machine learning model is trained to predict the half bandgap location of the energy level, and successfully overcome the traditional approach’s limitation. The proposed approach is validated using experimental measurements, where the machine learning predicts defect energy level and capture cross-section ratio within the uncertainty range of the traditional fitting method. The successful application of machine learning in the context of bulk defect parameter extraction paves the way to more complex data-driven physical models which have the potential to overcome the limitation of traditional approaches and can be applied to other materials such as perovskite and thin film.

Figure data from the paper can be found in NPJ Comput. Mater. Data folder

Cite as :
*Buratti, Y., Le Gia, Q. T., Dick, J., Zhu, Y. & Hameiri, Z. Extracting bulk defect parameters in silicon wafers using machine learning models. npj Computational Materials 6, 1–8 (2020)*

Related work:
*Buratti, Y., Dick, J., Gia, Q. L. & Hameiri, Z. A machine learning approach to defect parameters extraction: using random forests to inverse the Shockley-Read-Hall equation. in 46th IEEE Photovoltaic Specialist Conference 4 (2019)*


##  Installation
1/ Have Python 3.6+ installed (https://www.python.org/downloads/)

2/ Download the semiconductor package (https://github.com/MK8J/semiconductor)

3/ Move the semiconductor package in your Python\Python3X\Lib\site-packages\ (where you installed python)

4/ Download Github Repo folder and install the following package:
  * numpy
  * pandas
  * matplotlib
  * scipy
  * sklearn
  * pickle
  * ruamel_yaml

5/ Download examples files from this repo, or write your own!
