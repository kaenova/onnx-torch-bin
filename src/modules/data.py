import numpy as np

# This just a black box function that create 
# x and y. This function create linear function f(x) = 2x + 5
def data_generator():
    
    x = []
    y = []
    
    for i in range(100):
        x.append(i)
        y.append(2 * i + 5)
        
    x = np.asarray(x)
    y = np.asarray(y)
    
    return {
        "input" : x,
        "output" : y
    }