"Get attributes and print all attributes"
import pickle

def PrintSelf(obj):
    """Print out objects attribute"""
    for attr in dir(obj):
        if not attr.startswith("__"):
            value = getattr(obj, attr)
            if not callable(value):
                print(str(attr)+" : "+str(value))

def SaveObj(obj, folder, name):
    """Save python object using pickle"""
    if '.pkl' in name:
        with open(folder + name, 'rb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    else:
        with open(folder + name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def LoadObj(folder, name):
    """Load python object using pickle"""
    if '.pkl' in name:
        with open(folder + name, 'rb') as f:
            return pickle.load(f)
    else:
        with open(folder + name + '.pkl', 'rb') as f:
            return pickle.load(f)
def recur_items(dictionary,nest):
    for key, value in dictionary.items():
        if type(value) is dict:
            yield (key,nest)
            yield from recur_items(value,nest+1)
        else:
            yield (key,nest)

def print_dic(dic,nest=0):
    for key,nest in recur_items(dic,0):
        print("\t"*nest,key)
