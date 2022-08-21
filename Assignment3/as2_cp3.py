# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
from random import shuffle

"""# Functions"""

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def loadBatch(filename):
  for i in filename:
    features=unpickle(i)[b'data']
    labels=unpickle(i)[b'labels']
  X=features
  y=labels
  Y=one_hot_encode(10,y)
  return X,Y,y

def one_hot_encode(category,y):
  Y=np.eye(category)
  Y1=[]
  for ele in y:
    Y1.append(Y[int(ele)])
  return Y1
class Classifier():
  def __init__(self,X,Y,y,k):
    self.X=X
    self.Y=Y
    self.y=y
    self.k=k
    self.W=[]
    self.b=[]
    self.activation=[]
  def addDenseLayer(self,shape,activation):
    w=np.random.normal(0, 1/np.sqrt(shape[1]), (shape[0],shape[1]))
    b=np.zeros([shape[0],1])
    self.W.append(w)
    self.b.append(b)
    self.activation.append(activation)

  def evaluateClassifier(self,X):
    H=[]
    s=np.copy(np.transpose(X))
    n_b=tf.ones([1,X.shape[0]],tf.float64)
    for i in range(self.k):
      s=tf.matmul(self.W[i],s)+tf.matmul(self.b[i],n_b)
      if self.activation[i]=="relu":
        s=tf.nn.relu(s)
      elif self.activation[i]=="softmax":
        p=tf.nn.softmax(s,0)
      H.append(s)
    return H,p

  def computeCost(self,X,Y,lmd):
    H,P=self.evaluateClassifier(X)
    J_2=0
    n=X.shape[0]#X 50000,3072
    J_1_mat=-np.sum(tf.transpose(Y)*np.log(P))
    loss=J_1_mat/n
    for i in self.W:
      J_2=J_2+tf.math.reduce_sum(tf.square(i))
    cost=loss+lmd*J_2
    return loss,cost
  def accuracy(self,X,y):
    H,P=self.evaluateClassifier(X)
    predics=np.argmax(P,axis=0)
    a=predics-y
    count=0
    for i in a:
      if i==0:
        count+=1
    acc=count/len(y)
    return acc
  def Wb_gradient(self,X,Y,lmd):
    H,P=self.evaluateClassifier(X)
    n_b=X.shape[0]
    J_W=[]
    J_b=[]
    G_batch=-(tf.transpose(Y)-P)
    for l in range(self.k-1,0,-1):
     
      J_W_i=(1/n_b)*tf.matmul(G_batch,tf.transpose(H[l-1]))
      J_W.append(J_W_i)

      J_b_i=(1/n_b)*tf.matmul(G_batch,tf.ones([n_b,1],tf.float64))
      J_b.append(J_b_i)

      G_batch=tf.matmul(tf.transpose(self.W[l]),G_batch)
      G_batch=G_batch*self._Indicator(np.array(H[l-1]))
      
    J_W_1=(1/n_b)*tf.matmul(G_batch,X)
    J_b_1=(1/n_b)*tf.matmul(G_batch,tf.ones([n_b,1],tf.float64))
    J_W.append(J_W_1)
    J_b.append(J_b_1)

    J_W.reverse()
    J_b.reverse()

    return J_W,J_b
  @staticmethod
  def _Indicator(X):
    X[X>0]=1
    X[X<0]=0
    return X
  @staticmethod
  def generate_eta(eta_min,eta_max,n_s,t):
    k=(eta_max-eta_min)/(n_s)
    current_cycle = int(t / (2 * n_s))
    if 2*n_s*current_cycle<=t<=n_s+(2*n_s*current_cycle):
      eta_current=eta_min+k*(t-current_cycle*2*n_s)
    elif (current_cycle+1)*2*n_s>=t>=n_s+(2*n_s*current_cycle):
      eta_current=eta_max-k*(t-current_cycle*2*n_s-n_s)
    return eta_current
  def cyclic_Minibatch_gd(self,train_X,train_Y,train_y,epochs,batch_size,eta_min,eta_max,n_s,lmd,shuffle):
    n=int(train_X.shape[0]/batch_size)
    train_costs=[]
    validation_costs=[]
    train_accuracy=[]
    validation_accuracy=[]
   
    for i in tqdm(range(epochs)): 
      validation_acc=self.accuracy(validation_X,validation_y)
      train_acc=self.accuracy(train_X,train_y)
      _,validation_cost=self.computeCost(validation_X,validation_Y,lmd)
      _,train_cost=self.computeCost(train_X,train_Y,lmd)
      train_costs.append(train_cost)
      train_accuracy.append(train_acc)
      validation_costs.append(validation_cost)
      validation_accuracy.append(validation_acc)
      print("\n---training accuracy: ",train_acc,"---validation accuracy: ",validation_acc,"\n")
      print("---training loss: ",train_cost,"---validation loss: ",validation_cost,"\n")
      if i>epochs/3:
        eta_max=eta_max*0.1
      for j in range(n):
        t=i*n+j
        eta=self.generate_eta(eta_min,eta_max,n_s,t)
        j_start=j*batch_size
        j_end=(j+1)*batch_size
        Xtr=train_X[j_start:j_end]
        Ytr=train_Y[j_start:j_end]
        ytr=train_y[j_start:j_end]
        J_W,J_b=self.Wb_gradient(Xtr,Ytr,lmd)
        for l in range(self.k):
          self.W[l]=self.W[l]-eta*J_W[l]
          self.b[l]=self.b[l]-eta*J_b[l]
     
   
   #   if shuffle==True:
   #     tf.random.shuffle(train_X, seed=400)
   #     tf.random.shuffle(train_Y, seed=400)
   #     tf.random.shuffle(train_y, seed=400)
   
   
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    x=range(epochs)
    y1=train_costs
    y2=validation_costs
    ax.plot(x,y1, label='train')
    ax.plot(x, y2, label='validation')
    ax.set_xlabel('epochs')
    ax.set_ylabel('loss')
    ax.set_title('Cost')
    ax.legend()
    plt.show()
    fig, ax = plt.subplots()

    x=range(epochs)
    y1=train_accuracy
    y2=validation_accuracy
    ax.plot(x,y1, label='train')
    ax.plot(x,y2, label='validation')
    ax.set_xlabel('epochs')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy')
    ax.legend()
    plt.show()
    with open("train_cost.txt","w") as train_cost_file:
      train_cost=str(train_costs)
      train_cost_file.write(train_cost)
    with open("validation_cost.txt","w") as validation_cost_file:
      validation_cost=str(validation_costs)
      validation_cost_file.write(validation_cost)
    with open("train_accuracy.txt","w") as train_accuracy_file:
      train_accuracy=str(train_accuracy)
      train_accuracy_file.write(train_accuracy)
    with open ("validation_accuracy.txt","w") as validation_accuracy_file:
      validation_accuracy=str(validation_accuracy)
      validation_accuracy_file.write(validation_accuracy)    

  def minibatch(self,train_X,train_Y,train_y,n_batch,eta,lmd,shuffle):
    if shuffle==False:
      n=int(train_X.shape[0]/n_batch)
      for i in range(n):
        j_start=i*n_batch
        j_end=(i+1)*n_batch
        Xtr=train_X[j_start:j_end]
        Ytr=train_Y[j_start:j_end]
        ytr=train_y[j_start:j_end]

        J_W,J_b=self.Wb_gradient(Xtr,Ytr,lmd)
        for i in range(len(J_W)):
          self.W[i]=self.W[i]-eta*J_W[i]
          self.b[i]=self.b[i]-eta*J_b[i]
    elif shuffle==True:
      n=int(train_X.shape[0]/n_batch)
      for i in range(n):
        j_start=i*n_batch
        j_end=(i+1)*n_batch
        Xtr=train_X[j_start:j_end]
        Ytr=train_Y[j_start:j_end]
        ytr=train_y[j_start:j_end]

        J_W,J_b=self.Wb_gradient(Xtr,Ytr,lmd)
        for i in range(len(J_W)):
          self.W[i]=self.W[i]-eta*J_W[i]
          self.b[i]=self.b[i]-eta*J_b[i]
      tf.random.shuffle(train_X, seed=400)
      tf.random.shuffle(train_Y, seed=400)
      tf.random.shuffle(train_y, seed=400)

file1=['/cfs/klemming/home/y/yuchenga/DD2424/cifar-10-batches-py/data_batch_1']
X_n,Y_n,y_n=loadBatch(file1)
files=[['/cfs/klemming/home/y/yuchenga/DD2424/cifar-10-batches-py/data_batch_2'],['/cfs/klemming/home/y/yuchenga/DD2424/cifar-10-batches-py/data_batch_3'],['/cfs/klemming/home/y/yuchenga/DD2424/cifar-10-batches-py/data_batch_4'],['/cfs/klemming/home/y/yuchenga/DD2424/cifar-10-batches-py/data_batch_5']]
for i in range(len(files)):
  X,Y,y=loadBatch(files[i])
  X_n=list(X_n)+list(X)
  Y_n=list(Y_n)+list(Y)
  y_n=list(y_n)+list(y)
print(np.size(X_n,0))

#file1=['/Users/gaogao/Desktop/DD2424/Assignment2/cifar-10-batches-py/data_batch_1']
#X_n,Y_n,y_n=loadBatch(file1)

X=tf.cast(X_n,dtype=tf.float64)
Y=tf.cast(Y_n,dtype=tf.float64)
y=tf.cast(y_n,dtype=tf.float64)

from numpy.core.fromnumeric import std
mean_X=tf.reduce_mean(X,1)
mean_X=tf.reshape(mean_X,[50000,1])
ones=tf.ones([1,3072],dtype=tf.float64)

mean_X=tf.matmul(mean_X,ones)
std_X=tf.math.reduce_std(X,1)
std_X=tf.reshape(std_X,[50000,1])
std_X=tf.matmul(std_X,ones)

X=tf.subtract(X,mean_X)
X_n=tf.divide(X,std_X)
Y_n=Y
y_n=y

train_X=X_n[0:45000]
train_Y=Y_n[0:45000]
train_y=y_n[0:45000]

validation_X=X_n[45000:50000]
validation_Y=Y_n[45000:50000]
validation_y=y_n[45000:50000]

classifier=Classifier(train_X,train_Y,train_y,k=3)
classifier.addDenseLayer(shape=[50,3072],activation="relu")
classifier.addDenseLayer(shape=[50,50],activation="relu")
classifier.addDenseLayer(shape=[10,50],activation="softmax")

import math
eta_max=2e-3
eta_min=0.000012
batch_size=100

n_s=2250
cycles=2
iterations=2*n_s*cycles
epochs=int(iterations/batch_size)

classifier.cyclic_Minibatch_gd(train_X,train_Y,train_y,epochs=epochs,batch_size=batch_size,eta_min=eta_min,eta_max=eta_max,n_s=n_s,lmd=0.005,shuffle=True)

file=['/cfs/klemming/home/y/yuchenga/DD2424/cifar-10-batches-py/test_batch']
Xte,Yte,yte=loadBatch(file)

Xte=tf.cast(Xte,dtype=tf.float64)
Yte=tf.cast(Yte,dtype=tf.float64)
yte=tf.cast(yte,dtype=tf.float64)

from numpy.core.fromnumeric import std
mean_Xte=tf.reduce_mean(Xte,1)
mean_Xte=tf.reshape(mean_Xte,[10000,1])
ones=tf.ones([1,3072],dtype=tf.float64)

mean_Xte=tf.matmul(mean_Xte,ones)
std_Xte=tf.math.reduce_std(Xte,1)
std_Xte=tf.reshape(std_Xte,[10000,1])
std_Xte=tf.matmul(std_Xte,ones)

Xte=tf.subtract(Xte,mean_Xte)
Xte=tf.divide(Xte,std_Xte)

acc=classifier.accuracy(Xte,yte)

print(acc)

