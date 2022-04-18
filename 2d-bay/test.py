import numpy as np



X = np.zeros([3,2])
X[0,0] = 0
X[1,0] = 1
X[2,0] = 2

X[0,1] = 3
X[1,1] = 4
X[2,1] = 5
# Y1 = X[:,0]*X[:,0]+X[:,1]+X[:,1]

# print("result is ",Y1)

#print("result is ",Y1)

#X = np.zeros([3,2])
idx = 0
print("len:", len(X))
for x1 in X:
    print("idx:",idx,": ",x1)