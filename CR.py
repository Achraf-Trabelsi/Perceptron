
from sklearn import datasets 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split



#Generating Data 
#fixing the mean and cov values
mu1,mu2=[-1,0],[1,0]
sigma1=float(input("ENTER THE 1st COV VALUE "))
sigma2=float(input("ENTER THE 2nd 1COV VALUE "))
#Generating half by half
X1,y1=datasets.make_gaussian_quantiles(mean=mu1,cov=sigma1,n_samples=125,n_features=2,n_classes=1)
X2,y2=datasets.make_gaussian_quantiles(mean=mu2,cov=sigma2,n_samples=125,n_features=2,n_classes=1)
y1=np.ones(shape=(125,))
y2=-y1
#linking the dataset and target 
X=np.concatenate((X1,X2),axis=0)
y=np.concatenate((y1,y2),axis=0)
#plotting the distribution
fig = plt.figure(figsize=(10,8)) 
plt.plot(X[:, 0][y == -1], X[:, 1][y ==-1], 'r^') 
plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], 'bs') 
plt.xlabel("feature 1") 
plt.ylabel("feature 2")
plt.title('Random Classification Data with 2 classes')
#plt.show()
#creating a test and train split
x_train,y_train,x_test,y_test=train_test_split(X,y,test_size=0.2,shuffle=True)



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
w,miss=perceptron(X,y,0.1,100)
print(w,miss)
plot_decision_boundary(X,w)
plt.show()
