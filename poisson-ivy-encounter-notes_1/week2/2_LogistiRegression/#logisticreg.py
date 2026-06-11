#logisticreg
#% Importing pandas
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

# create the sigmoid function
def my_function(f):
    '''Calculates p for given f'''
    # p = np.exp(-f)
    result = 1/(1+np.exp(-f))
    return result


f = np.arange(-50, 50, 1)  # creates numbers between -50 and 50 with a stepsize of 1
p = my_function(f)         # the 101 corresponding function results

plt.plot(f,p)
plt.xlabel('f')
plt.ylabel('p')
plt.title('The sigmoid function p')
# creating Sigmoid Function
def sigmoid_function(f):
    '''Calculates p for given f'''
    p = 1/(1+np.exp(-f))
    return p

# create the logit function
def logit_function(x, b, w):
    '''Caluculates the border'''
    f = b + w*x
    return f

b = 0
w = 1

x = np.arange(-50, 50, 1)
f = logit_function(x, b, w)

p = sigmoid_function(f)

plt.plot(x,p) 
plt.suptitle('The sigmoid function p')
plt.title(f'for b={b} and w={w}')
plt.show()