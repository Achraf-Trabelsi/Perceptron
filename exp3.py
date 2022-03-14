from sklearn import datasets 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib.pyplot import errorbar
err_tab=[]
sigma=[0.01, 0.1, 0.5, 0.7]
for s in sigma:
    #Generating Data 
    #fixing the mean and cov values
    mu1,mu2=[-1,0],[1,0]
    sig=[0.01, 0.1, 0.5, 0.7]
    #Generating half by half
    X1,y1=datasets.make_gaussian_quantiles(mean=mu1,cov=s,n_samples=125,n_features=2,n_classes=1)
    X2,y2=datasets.make_gaussian_quantiles(mean=mu2,cov=s,n_samples=125,n_features=2,n_classes=1)
    y1=np.ones(shape=(125,))
    y2=-y1
    #linking the dataset and target 
    X=np.concatenate((X1,X2),axis=0)
    y=np.concatenate((y1,y2),axis=0)
    x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,shuffle=True)
    print(x_train.shape,x_test.shape)

    def acti_func(z):
      return 1 if z>-0.5 else -1
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
    w,miss=perceptron(x_train,y_train,0.1,200)
    print(w,miss)

    #Ploting the decision boundry
    plt.plot(x_train[:, 0][y_train == -1], x_train[:, 1][y_train==-1], 'r^') 
    plt.plot(x_train[:, 0][y_train == 1], x_train[:, 1][y_train == 1], 'bs') 
    #plot_decision_boundary(x_train,w)
    plt.ylim([ min(x_train[:, 1]), max(x_train[:, 1]) ])
    plt.show()

    #Testing the model
    plt.plot(x_test[:, 0][y_test == -1], x_test[:, 1][y_test==-1], 'r^') 
    plt.plot(x_test[:, 0][y_test == 1], x_test[:, 1][y_test == 1], 'bs') 
    #plot_decision_boundary(x_test,w)
    plt.ylim([ min(x_train[:, 1]), max(x_train[:, 1]) ])
    plt.show()

    #Computing the accuracy of the model
    print("the accuracy of the model on the test set: ",(1-(miss[len(miss)-1]/200))*100)
    err=(miss[len(miss)-1]/200)*100
    
    err_tab.append(err)
print(err_tab)
print("la moyenne est =",np.mean(err_tab),"la varaince est",np.var(err_tab))
errorbar(sig,err_tab, marker='s', mfc='red',
         mec='green', ms=10, mew=2)
plt.show()



