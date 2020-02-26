"Get attributes and print all attributes"
import pickle

def PrintSelf(obj):
    for attr in dir(obj):
        if not attr.startswith("__"):
            value = getattr(obj, attr)
            if not callable(value):
                print(str(attr)+" : "+str(value))

def SaveObj(obj, folder, name):
    if '.pkl' in name:
        with open(folder + name, 'rb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    else:
        with open(folder + name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def LoadObj(folder, name):
    if '.pkl' in name:
        with open(folder + name, 'rb') as f:
            return pickle.load(f)
    else:
        with open(folder + name + '.pkl', 'rb') as f:
            return pickle.load(f)
