import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sympy import Symbol

### x->y: x is 1D, y is 1D.

###################################################################################################################################
###########################Required Class for Gaussian Process Regression##########################################################
###################################################################################################################################
"""
function that simulates the blackbox function
"""
def blackBoxFcn(X):
    # the function, which is y = x^2 here
    #y = X*np.sin(X**2)-1*np.cos(X**2)
    #y = X*np.cos(X**2)
    #y = X*np.sin(X**4)
    Y = np.zeros([X.shape[0]])
    for i in range(X.shape[0]):
        if(X[i]<=0.03):
            Y[i] = 1
        if(X[i]>0.03):
            Y[i] = 1-np.tanh(10*X[i])
    return Y

"""
function that simulates the blackbox function
"""
def blackBoxFcn2(X):
    # the function, which is y = x^2 here
    #y = X*np.sin(X**2)-1*np.cos(X**2)
    #y = X*np.cos(X**2)
    #y = X*np.sin(X**4)
    Y = np.zeros([X.shape[0]])
    for i in range(X.shape[0]):
        if(X[i]<=0.03):
            Y[i] = 1
        if(X[i]>0.03):
            Y[i] = 0
    return Y


"""
Ground truth function
100 linearly spaced numbers used to sample the ground-truth blackbox function, only used as reference
"""
x = np.linspace(0.005,0.1,120)

# the function, which is y = x^2 here
#y = x**2
#y = x*np.sin(x**2)
#y = x*np.cos(x**2)
y1 = blackBoxFcn(x)
y2 = blackBoxFcn2(x)

#draw 
# # setting the axes at the centre
fig = plt.figure()
fig.set_size_inches(16,12)
ax = fig.add_subplot(1, 1, 1)
plt.xlim([0, 0.2])
plt.ylim([-5, 5])
ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position('zero')

# ax.spines['left'].set_position('zero')
# ax.spines['bottom'].set_position('zero')

ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')


"""
plot blackbox ground truth
"""
plt.plot(x,y1,linestyle = '--', color ='gray', linewidth = 1.0)
plt.plot(x,y2,linestyle = '--', color ='red', linewidth = 1.0)

plt.show()