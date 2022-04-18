import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sympy import Symbol
from matplotlib import cm
from matplotlib.ticker import LinearLocator
### x->y: x is 2D, y is 1D.

###################################################################################################################################
###########################Required Class for Gaussian Process Regression##########################################################
###################################################################################################################################
"""
function that simulates the blackbox function
used in calculation, not in visualization
@X -> nx2 
@Y -> nx1
"""
def blackBoxFcn(X):
    # the function, which is y = x^2 here
    #y = X*np.sin(X**2)-1*np.cos(X**2)
    #y = X*np.cos(X**2)
    #y = X*np.sin(X**4)
    
    # Y = X[:,0]*X[:,0]+X[:,1]*X[:,1]
    Y = X[:,0]*X[:,0]+X[:,1]*X[:,1]-2
    return Y

"""
class used to draw matplotlib 3d surface
used in visualization, not in calculation
X1 -> first input parameter 
X2 -> second input parameter
"""
paraB = 1.5
paraA = 2.0
def blackBoxFcnMeshGrid(X1, X2):

    Y = np.zeros([X1.shape[0],X1.shape[1]],dtype= float)
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            # Y[i,j] = X1[i,j]*X1[i,j] + X2[i,j]*X2[i,j]
            Y[i,j] = X1[i,j]*X1[i,j] + X2[i,j]*X2[i,j] - 2
    return Y

"""
class that calculates the kernel function
"""
class SquaredExponentialKernel:
    def __init__(self, sigma_f: float = 1, length: float = 1):
        self.sigma_f = sigma_f
        self.length = length

    #calculate covariance
    
    def __call__(self, argument_1: np.array, argument_2: np.array) -> float:
        return float(self.sigma_f *
                     np.exp(-(np.linalg.norm(argument_1 - argument_2)**2) /
                            (2 * self.length**2)))

"""
function that calculates the respective covariance matrices
2d version: x1 -> nx2; x2 -> mx2
"""
def cov_matrix(x1, x2, cov_function) -> np.array:
    return np.array([[cov_function(a, b) for a in x1] for b in x2])

"""
class that encapsulates the Gaussian Process Regression
"""
class GPR:
    def __init__(self):
        pass

    #fit surrogate model using dataset
    def updateRegressor(self,
                        data_x: np.array,
                        data_y: np.array,
                        covariance_function=SquaredExponentialKernel(),
                        white_noise_sigma: float = 0):
        self.noise = white_noise_sigma
        self.data_x = data_x
        self.data_y = data_y
        self.covariance_function = covariance_function

        # Store the inverse of covariance matrix of input (+ machine epsilon on diagonal) since it is needed for every prediction
        self._inverse_of_covariance_matrix_of_input = np.linalg.inv(
            cov_matrix(data_x, data_x, covariance_function) +
            (3e-7 + self.noise) * np.identity(len(self.data_x)))             
        

    # function to predict output at new input values. Store the mean and covariance matrix in memory.
    def predict(self, at_values: np.array) -> np.array:
        print("Predict starts")
        k_lower_left = cov_matrix(self.data_x, at_values,
                                  self.covariance_function)
        k_lower_right = cov_matrix(at_values, at_values,
                                   self.covariance_function)
        print("Predict 1")
        # Mean.
        mean_at_values = np.dot(
            k_lower_left,
            np.dot(self.data_y,
                   self._inverse_of_covariance_matrix_of_input.T).T).flatten()

        print("Predict 2")
        # Covariance.
        cov_at_values = k_lower_right - \
            np.dot(k_lower_left, np.dot(
                self._inverse_of_covariance_matrix_of_input, k_lower_left.T))

        print("Predict 3")
        # Adding value larger than machine epsilon to ensure positive semi definite
        cov_at_values = cov_at_values + 3e-7 * np.ones(
            np.shape(cov_at_values)[0])

        print("Predict 4")
        var_at_values = np.diag(cov_at_values)

        self._memory = {
            'mean': mean_at_values,
            'covariance_matrix': cov_at_values,
            'variance': var_at_values
        }

"""
class that encapsulates the acquisition function
here, we implement Upper Confidence Bound (UCB)
"""
class GPAF:
    def __init__(self, lamda=1.0):
        self.lamda = lamda
    
    def updateParameter(self, lamda = 1.0):
        self.lamda = lamda
    
    #calculate extreme value of acquisition function:
    #return a x value
    def calMinimum(self, at_values, mean_at_values, div_at_values):
        SortingArray = mean_at_values - self.lamda * div_at_values
        minimumValue = np.amin(SortingArray)
        manIdx = np.where(SortingArray == minimumValue)
        return at_values[manIdx]

"""
class that encapsulates Bayesian Optimizer
"""
class BayesianOptimizer:
    def __init__(self, GaussianRegressor = GPR(), accquisitionFcn = GPAF()):
        self.regressor = GaussianRegressor
        self.accquisitionFcn = accquisitionFcn
    
    # def updateKernelPara(self, sigma_f, length):
        # self.regressor = 
    
    #fit surrogateModel using dataset
    #remember, you can choose any kernel parameter you want. If you dont indicate, there will be default value -> sigma_f: float = 1, length: float = 1
    #the way you 
    def fitSurrogateModel(self,
                          data_x: np.array,
                          data_y: np.array,
                          covariance_function=SquaredExponentialKernel(),
                          white_noise_sigma: float = 0):
        self.regressor.updateRegressor(data_x, data_y, covariance_function, white_noise_sigma)
    
    def predict(self, at_values):
        self.regressor.predict(at_values)
    
    def getPredictedMeanCov(self):
        self.mean = self.regressor._memory['mean']
        self.var = self.regressor._memory['variance']
        self.stdDev = np.sqrt(self.var)
        self.covMat = self.regressor._memory['covariance_matrix']
     
    
    #use acquisition function to get the new sample points
    def callAcqusitionFcn(self, at_values, mean_at_values, div_at_values, lamda):
        self.accquisitionFcn.updateParameter(lamda)
        return self.accquisitionFcn.calMinimum(at_values, mean_at_values, div_at_values)
###################################################################################################################################
#################################################Main working flow#################################################################
###################################################################################################################################


"""
Ground truth function
100 linearly spaced numbers used to sample the ground-truth blackbox function, only used as reference
"""
num = 20

#our data formate
X = np.zeros([num*num, 2],dtype =float)

xStart = -2.0
xEnd = 2.0
xInterval = (xEnd - xStart)/(num-1)

yStart = -2
yEnd = 2
yInterval = (yEnd - yStart)/(num-1)


idx = 0
for i in range(num):
    for j in range(num):
        X[idx][0] = xStart + i * xInterval
        X[idx][1] = yStart + j * yInterval
        idx = idx + 1


#use to call matplotlib function
drawX1 = np.arange(xStart, xEnd+xInterval, xInterval)
drawX2 = np.arange(yStart, yEnd+yInterval, yInterval)

#draw ground truth surface
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
drawX1, drawX2 = np.meshgrid(drawX1,drawX2)
drawY = blackBoxFcnMeshGrid(drawX1,drawX2)
ax.set_xlabel('x')
ax.set_ylabel('y')
surf = ax.plot_surface(drawX1, drawX2, drawY, alpha=0.2,
                       linewidth=0, antialiased=False)
# plt.show()


"""
Initialize Bayesian Optimizer with indicated kernel parameters
"""
kernel = SquaredExponentialKernel(sigma_f=0.5, length=0.5)
#kernel = SquaredExponentialKernel(sigma_f=1.0, length=1.0)
bOptimizer = BayesianOptimizer()

"""
Zero, initial randomly sample serveral points as dataset
First, fit a surrogate model (gaussian regression model) using sampled dataset
Second, use acquisition function to guide the next few sample points
Third, append the newly selected sample points into the dataset
Fourth, back to first step until a threshold is reached
"""
##random number genreation
#print(np.random.uniform(5,10,[3,3]))
#print(np.random.uniform(5,10,[6]))

##initial data generation
# initX = np.random.uniform(-2,2,[5,2])

initX = np.random.uniform(-2,2,[5,2])

initX[0][0] = -1.524
initX[1][0]= -0.5123
initX[2][0] = 1.25623
initX[3][0] = -0.47407809
initX[4][0] = -1.98444269

initX[0][1] = 1.25623
initX[1][1] = -0.47407809
initX[2][1] = -1.98444269
initX[3][1] = 0.47407809
initX[4][1] = 1.98444269

initY = blackBoxFcn(initX)

##assign to be dataset
data_X = initX
data_Y = initY

# ###########
#iteration
maximumIter = 15
for i in range(0,maximumIter):
    ##fit the gaussian model
    bOptimizer.fitSurrogateModel(data_X, data_Y, kernel)

    ##predict the model value at given points
    predictX = X
    bOptimizer.predict(predictX)
    
    
    bOptimizer.getPredictedMeanCov()

    predictMean = bOptimizer.mean
    predictVar = bOptimizer.var
    predictStdDev = bOptimizer.stdDev

    predictUpperBound = predictMean+predictStdDev
    predictLowerBound = predictMean-predictStdDev


    ##utilize acquisition function to decide the newly added sample points
    newPnt = bOptimizer.callAcqusitionFcn(predictX, predictMean, predictStdDev, 1.5)

    
    if i != maximumIter-1:
        #print("Original    X: ",data_X)
        #print("newPnt       : ",newPnt)
        data_X = np.concatenate((data_X, newPnt), axis=0)
        #print("Newly added X: ",data_X)
        data_Y = np.append(data_Y, blackBoxFcn(newPnt))
        
        
        
# ###################################################################################################################################
# #####################################################Visualization#################################################################
# ###################################################################################################################################


# #draw 
# # # setting the axes at the centre
# fig = plt.figure()
# fig.set_size_inches(16,12)
# ax = fig.add_subplot(1, 1, 1)
# plt.xlim([-2.5, 2.5])
# plt.ylim([-5, 5])
# ax.spines['left'].set_position('center')
# ax.spines['bottom'].set_position('zero')

# # ax.spines['left'].set_position('zero')
# # ax.spines['bottom'].set_position('zero')

# ax.spines['right'].set_color('none')
# ax.spines['top'].set_color('none')
# ax.xaxis.set_ticks_position('bottom')
# ax.yaxis.set_ticks_position('left')

# """
# upper bound
# """
# ax.scatter3D(predictX[:,0], predictX[:,1], predictUpperBound, 'black',s = 10)

# """
# lower bound
# """
# ax.scatter3D(predictX[:,0], predictX[:,1], predictLowerBound, 'red',s = 10)

fig.set_size_inches(16,12)

"""
prediced mean
"""
##mean's scatter point
ax.scatter3D(predictX[:,0], predictX[:,1], predictMean, 'red',s = 10)

##mean's surface: continuous representation
visualizedPredictedSurfaceVal = np.transpose(predictMean.reshape((num,num)))
ax.plot_surface(drawX1, drawX2, visualizedPredictedSurfaceVal, alpha=0.5, color = 'black',
                       linewidth=0, antialiased=False)
             

             
"""
shade area between two bounds
"""
visualizedPredictedSurfaceLowerBoundVal = np.transpose(predictUpperBound.reshape((num,num)))
ax.plot_surface(drawX1, drawX2, visualizedPredictedSurfaceLowerBoundVal, alpha=0.3, color = 'red',
                       linewidth=0, antialiased=False)

visualizedPredictedSurfaceHighBoundVal = np.transpose(predictLowerBound.reshape((num,num)))
ax.plot_surface(drawX1, drawX2, visualizedPredictedSurfaceHighBoundVal, alpha=0.3, color = 'blue',
                       linewidth=0, antialiased=False)



"""
plot scatter ground truth points
"""
ax.scatter3D(data_X[:,0], data_X[:,1], data_Y, color = 'red',s = 60)

# show the plot

elev = 4
azim = -65
ax.view_init(elev, azim)
plt.show()