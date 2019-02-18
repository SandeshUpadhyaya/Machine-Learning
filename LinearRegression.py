import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#Initializin the values of X and Y
X = np.random.rand(100,1)
Y = np.random.rand(100,1)
plt.scatter(X,Y)
plt.show()

#Adding one to X
X = np.concatenate((np.ones((100,1)),X),axis=1)

#splitting data into train and test set
X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=1)

#defining the hypothesis
def hypothesis(x,w):
  return np.dot(x,w)

#defining the error
def error(x,y,w):
  return (hypothesis(x,w) - y)

#defining the cost function
def computeCost(x,y,w):
  m = len(y)
  J = (1/(2*m) * np.sum(error(x,y,w)**2)) 
  return J

#Initializing the weights 
w = np.random.rand(X_train.shape[1]).reshape(2,1)

#hypothesis for data before training
plt.scatter(X_train[0:,1:], y_train,color = "m", marker = "o", s = 30) 
# predicted response vector 
y_pred = hypothesis(X_train,w) 
# plotting the regression line 
plt.plot(X_train[0:,1:], y_pred, color = "g") 
# putting labels 
plt.xlabel('x') 
plt.ylabel('y') 
# function to show plot 
plt.show() 

#gradient descent implementation
def gradientDescent(x,y,w,lr,iter):
  train_loss = []
  m = len(y)
  for i in range(iter):
    w = w - ((lr/m) * np.sum(error(x,y,w)*x))
    J = computeCost(x,y,w)
    train_loss.append(J)
  plt.plot(range(iter),train_loss)
  plt.show()
  print("Starting training loss: ",train_loss[0],"Last training loss ",train_loss[-1])
  print("Train loss is ",train_loss[0]-train_loss[-1])
  return w

#gradient descent output
w1 = gradientDescent(X_train,y_train,w,0.01,700)