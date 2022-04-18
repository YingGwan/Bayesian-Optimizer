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
    return X*np.sin(X**2)-1*np.cos(X**2)

"""
class that calculates the kernel function
"""
class SquaredExponentialKernel:
    def __init__(self, sigma_f: float = 1, length: float = 1):
        self.sigma_f = sigma_f
        self.length = length

    def __call__(self, argument_1: np.array, argument_2: np.array) -> float:
        return float(self.sigma_f *
                     np.exp(-(np.linalg.norm(argument_1 - argument_2)**2) /
                            (2 * self.length**2)))

"""
function that calculates the respective covariance matrices
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
        k_lower_left = cov_matrix(self.data_x, at_values,
                                  self.covariance_function)
        k_lower_right = cov_matrix(at_values, at_values,
                                   self.covariance_function)

        # Mean.
        mean_at_values = np.dot(
            k_lower_left,
            np.dot(self.data_y,
                   self._inverse_of_covariance_matrix_of_input.T).T).flatten()

        # Covariance.
        cov_at_values = k_lower_right - \
            np.dot(k_lower_left, np.dot(
                self._inverse_of_covariance_matrix_of_input, k_lower_left.T))

        # Adding value larger than machine epsilon to ensure positive semi definite
        cov_at_values = cov_at_values + 3e-7 * np.ones(
            np.shape(cov_at_values)[0])

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
x = np.linspace(-2,2,120)

# the function, which is y = x^2 here
#y = x**2
#y = x*np.sin(x**2)
#y = x*np.cos(x**2)
y = blackBoxFcn(x)

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
initX = np.random.uniform(-2,2,[3])
initX[0] = -1.524
initX[1] = -0.5123
initX[2] = 1.25623


initY = blackBoxFcn(initX)

##assign to be dataset
data_X = initX
data_Y = initY


###########
#iteration
maximumIter = 10
for i in range(0,maximumIter):
    ##fit the gaussian model
    bOptimizer.fitSurrogateModel(data_X, data_Y, kernel)

    ##predict the model value at given points
    predictX = np.linspace(-2,2,150)
    bOptimizer.predict(predictX)
    bOptimizer.getPredictedMeanCov()

    predictMean = bOptimizer.mean
    predictVar = bOptimizer.var
    predictStdDev = bOptimizer.stdDev

    predictUpperBound = predictMean+predictStdDev
    predictLowerBound = predictMean-predictStdDev


    ##utilize acquisition function to decide the newly added sample points
    newPnt = bOptimizer.callAcqusitionFcn(predictX, predictMean, predictStdDev, 1.0)

    
    if i != maximumIter-1:
        data_X = np.append(data_X, newPnt)
        data_Y = np.append(data_Y, blackBoxFcn(newPnt))
        print("Newly added X: ",data_X)
        print("Newly added Y: ",data_Y)
        
###################################################################################################################################
#####################################################Visualization#################################################################
###################################################################################################################################


#draw 
# # setting the axes at the centre
fig = plt.figure()
fig.set_size_inches(16,12)
ax = fig.add_subplot(1, 1, 1)
plt.xlim([-2.5, 2.5])
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
shade area between two bounds
"""
ax.fill_between(predictX, predictLowerBound, predictUpperBound, color = 'mediumslateblue')

"""
plot blackbox ground truth
"""
plt.plot(x,y,linestyle = '--', color ='gray', linewidth = 3.0)

"""
plot predicted mean
"""
plt.plot(predictX, predictMean, color = 'black', linewidth = 2.0)


"""
plot scatter ground truth points
"""
plt.scatter(data_X, data_Y, color = 'red',s = 35)

# show the plot
plt.show()