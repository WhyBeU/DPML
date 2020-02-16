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
#---    Class functionalities and definition
#///////////////////////////////////////////
# %%--  Class functionalities and definition
def checkCell():
    print("Test - Cell class")
    cell = Cell()
    T = np.random.randint(low = Cell.Tmin, high=Cell.Tmax)
    if cell.changeT(T).T!=T: print("Cell.changeT error")
    Ndop = np.exp(np.random.randint(low = np.log(Cell.Ndopmin), high=np.log(Cell.Ndopmax)))
    if cell.changeNdop(Ndop).Ndop!=Ndop: print("Cell.changeNdop error")
    print("Test complete")
def checkDefect():
    print("Test - Defect class")
    N = np.random.randint(100)
    db = Defect.random_db(N)
    if len(db)!=N: print("Defect.random_db error")
    print("Test complete")
# %%-

# %%--  Run test function
checkCell()
checkDefect()

# %%-
