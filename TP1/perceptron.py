
from sklearn import datasets 
import matplotlib.pyplot as plt
import numpy as np

X, y = datasets.make_blobs (n_samples=150, n_features=2, centers=2, 
cluster_std=1.05, random_state=2) 
#Plotting 
fig = plt.figure(figsize=(10,8)) 
plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], 'r^') 
plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], 'bs') 
plt.xlabel("feature 1") 
plt.ylabel("feature 2")
plt.title('Random Classification Data with 2 classes')


def acti_func(z):
    return 1 if z>0 else 0

def perceptron(X, y, lr, epochs): 
    # X --> Inputs. 
    # y --> labels/target. 
    # lr --> learning rate. 
    # epochs --> Number of iterations. 
    # m-> number of training examples 
    # n-> number of features 
    m,n = X.shape 
    # Initializing parameters(theta) to zeros. 
    # +1 in n+1 for the bias term. 
    w = np.zeros((n+1,1)) 
    # Empty list to store how many examples were
    # misclassified at every iteration. 
    n_miss_list = [] 
    # Training. 
    for epoch in range(epochs):
    # variable to store #misclassified. 
      n_miss = 0 
      # looping for every example. 
      for idx, x_i in enumerate(X):
          # Insering 1 for bias, 
          X0 = 1 
          x_i = np.insert(x_i, 0, 1).reshape(-1,1) 
          # Calculating prediction/hypothesis. 
          y_hat = acti_func(np.dot(x_i.T, w)) 
          # Updating if the example is misclassified. 
          if (np.squeeze(y_hat) - y[idx]) != 0: 
              w=w-lr*(np.squeeze(y_hat) - y[idx])*x_i
            # Incrementing by 1. 
              n_miss+=1
          # Appending number of misclassified examples
          # at every iteration.
      n_miss_list.append(n_miss) 
    return w, n_miss_list

def plot_decision_boundary(X, w): 
# X --> Inputs 
# w --> parameters 
# The Line is y=mx+c
# So, Equate mx+c = w0.X0 + w1.X1 + w2.X2
# Solving we find m and c 
 x1 = [min(X[:,0]), max(X[:,0])] 
 m = -w[1]/w[2]
 c = -w[0]/w[2]
 x2 = m*x1 + c
# Plotting
 plt.plot(x1,x2)
 plt.show()
w,miss=perceptron(X,y,0.1,10)
print(perceptron(X,y,0.1,10))
print(plot_decision_boundary(X,w))
