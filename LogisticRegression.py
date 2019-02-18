import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

X = np.random.rand(1000,1)
Y = np.random.rand(1000,1)
plt.scatter(X,Y)
plt.show()

def hypothesis(x,w):
  return np.dot(x,w)

def sigmoid(x,w):
  return 1/(1+np.exp(-hypothesis(x,w)))

def cost(x,y,w):
  m = len(y)
  J = (-1/m)*(sum(y*np.log(sigmoid(x,w))+(1-y)*np.log(1-sigmoid(x,w))))
  return J

X = np.concatenate((np.ones((1000,1)),X),axis=1)

X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=1)

def gradient(x,y,w,lr,iter):
  train_loss = []
  for i in range(iter):
    w = w - ((lr) * np.dot(np.transpose(x),(sigmoid(x,w)-y)))
    J = cost(x,y,w)
    train_loss.append(J)
  plt.plot(range(iter),train_loss)
  plt.show()
  print("Starting training loss: ",train_loss[0],"Last training loss ",train_loss[-1])
  print("Train loss is ",train_loss[0]-train_loss[-1])

w = np.random.rand(X_train.shape[1]).reshape(2,1)

gradient(X_train,y_train,w,0.001,300)