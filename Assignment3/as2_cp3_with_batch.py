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
  def __init__(self,X,Y,y,k,batch_normalization):
    self.X=X
    self.Y=Y
    self.y=y
    self.k=k
    self.batch_normalization=batch_normalization
    self.W=[]
    self.b=[]
    self.beta=[]
    self.gamma=[]
    self.activation=[]
  def addDenseLayer(self,shape,activation):
    w,b,beta,gamma=self._initialization(shape)
    self.W.append(w)
    self.b.append(b)
    self.beta.append(beta)
    self.gamma.append(gamma)
    self.activation.append(activation)

  def _initialization(self,shape):
    w=np.random.normal(0, 1/np.sqrt(shape[1]), (shape[0],shape[1]))
    b=np.zeros([shape[0],1])
    beta=np.zeros([shape[0],1])
    gamma=np.ones([shape[0],1])

    return w,b,beta,gamma

  def evaluateClassifier(self,X):
    s=np.copy(np.transpose(X))
    n=X.shape[0]
    n_b=tf.ones([1,X.shape[0]],tf.float64)
    if self.batch_normalization==False:
      H=[]
      for i in range(self.k):
        s=tf.matmul(self.W[i],s)+tf.matmul(self.b[i],n_b)
        if self.activation[i]=="relu":
          s=tf.nn.relu(s)
        elif self.activation[i]=="softmax":
          p=tf.nn.softmax(s,0)
        H.append(s)
      return H,p
    else:
      S,S_hat,S_tilde,means,variance,H=[],[],[],[],[],[]
      H.append(s)
      for i in range(self.k-1):
        s=tf.matmul(self.W[i],s)+tf.matmul(self.b[i],n_b)
        S.append(s)
        mu=np.mean(s,axis=1,keepdims=True)
        means.append(mu)
        var = np.var(s, axis=1, keepdims=True) * (n-1)/n
        variance.append(var)
        s_hat = (s - mu) / np.sqrt(var + np.finfo(np.float).eps)
        S_hat.append(s_hat)
        s_tilde=np.multiply(tf.matmul(self.gamma[i],n_b),s_hat)+tf.matmul(self.beta[i],n_b)
        if self.activation[i]=="relu":
          s=tf.nn.relu(s_tilde)
        H.append(s)
      p=tf.matmul(self.W[self.k-1],s)+tf.matmul(self.b[self.k-1],n_b)
      p=tf.nn.softmax(p,0)
      H.append(p)
      return S,S_hat,means,variance,H

  def computeCost(self,X,Y,lmd):
    if self.batch_normalization==True:
       _,_,_,_,H=self.evaluateClassifier(X)
       P=H[-1]
    else:
      H,P=self.evaluateClassifier(X)
    J_2=0
    n=X.shape[0]
    J_1_mat=-np.sum(tf.transpose(Y)*np.log(P))
    loss=J_1_mat/n
    for i in self.W:
      J_2=J_2+tf.math.reduce_sum(tf.square(i))
    cost=loss+lmd*J_2
    return loss,cost

  def accuracy(self,X,y):
    if self.batch_normalization==True:
      _,_,_,_,H=self.evaluateClassifier(X)
      P=H[-1]

      predics=np.argmax(P,axis=0)
      a=predics-y
      count=0
      for i in a:
        if i==0:
          count+=1
      acc=count/len(y)
      return acc
    else:
      H,P=self.evaluateClassifier(X)
      predics=np.argmax(P,axis=0)
      a=predics-y
      count=0
      for i in a:
        if i==0:
          count+=1
      acc=count/len(y)
      return acc

  def gradient(self,X,Y,lmd):
    if self.batch_normalization==False:
      H,P=self.evaluateClassifier(X)
      n_b=X.shape[0]
      J_W=[]
      J_b=[]
      G_batch=-(tf.transpose(Y)-P)
      for i in range(self.k-1,0,1):
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

    elif self.batch_normalization==True:
      S,S_hat,means,variance,H=self.evaluateClassifier(X)
      n_b=X.shape[0]
      J_W=[]
      J_b=[]
      J_gamma=[]
      J_beta=[]
      G_batch=-(tf.transpose(Y)-H[-1])#10 100
      J_W_k=(1/n_b)*tf.matmul(G_batch,tf.transpose(H[self.k-1]))+2*lmd*self.W[self.k-1]
      J_b_k=(1/n_b)*tf.matmul(G_batch,tf.ones([n_b,1],tf.float64))
      J_W.append(J_W_k)
      J_b.append(J_b_k)

      G_batch=tf.matmul(tf.transpose(self.W[self.k-1]),G_batch)
      G_batch=G_batch*self._Indicator(np.array(H[self.k-1]))
      for l in range(self.k-2,-1,-1):
        J_gamma_l=(1/n_b)*tf.matmul((G_batch*S_hat[l]),tf.ones([n_b,1],tf.float64))
        J_beta_l=(1/n_b)*tf.matmul(G_batch,tf.ones([n_b,1],tf.float64))
        J_gamma.append(J_gamma_l)
        J_beta.append(J_beta_l)
        G_batch=G_batch*(tf.matmul(self.gamma[l],tf.transpose(tf.ones([n_b,1],tf.float64))))

        G_batch=self.BatchNormBackPass(G_batch,S[l],means[l],variance[l])

        J_W_l=(1/n_b)*tf.matmul(G_batch,tf.transpose(H[l]))+2*lmd*self.W[l]
        J_b_l=(1/n_b)*tf.matmul(G_batch,tf.ones([n_b,1],tf.float64))

        J_W.append(J_W_l)
        J_b.append(J_b_l)
        if l>0:
          G_batch=tf.matmul(tf.transpose(self.W[l]),G_batch)
          G_batch=G_batch*self._Indicator(np.array(H[l]))
      J_W.reverse()
      J_b.reverse()
      J_gamma.reverse()
      J_beta.reverse()
      return J_W,J_b,J_gamma,J_beta

  def BatchNormBackPass(self,G_batch,S,mu,var):
    sigma_1 = np.power(var + np.ones([G_batch.shape[0],1])*np.finfo(float).eps, -0.5)
    sigma_2 = np.power(var + np.ones([G_batch.shape[0],1])*np.finfo(float).eps, -1.5)
        
    g1 = G_batch * tf.matmul(sigma_1, tf.ones([1,G_batch.shape[1]],tf.float64))#30,8000
    g2 = G_batch *tf.matmul(sigma_2, tf.ones([1,G_batch.shape[1]],tf.float64))#30,8000
    d = S [1]- mu[1]#30,8000
    c = tf.matmul((g2*d),tf.ones([G_batch.shape[1],1],tf.float64))#30,1
    g_batch=g1-(1/G_batch.shape[1])*(tf.matmul(tf.matmul(g1,tf.ones([G_batch.shape[1],1],tf.float64)),tf.ones([1,G_batch.shape[1]],tf.float64)))-(1/G_batch.shape[1])*d*tf.matmul(c,tf.ones([1,G_batch.shape[1]],tf.float64))
    return g_batch


  @staticmethod
  def _Indicator(X):
    X[X>0]=1
    X[X<0]=0
    return X


  @staticmethod
  def generate_eta(eta_min,eta_max,n_s,t):
    k=(eta_max-eta_min)/(n_s)
    current_cycle=int(t/(2*n_s))
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
    if self.batch_normalization==False:
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
        
        for j in range(n):
          t=i*n+j
          eta=self.generate_eta(eta_min,eta_max,n_s,t)
          j_start=j*batch_size
          j_end=(j+1)*batch_size
          Xtr=train_X[j_start:j_end]
          Ytr=train_Y[j_start:j_end]
          ytr=train_y[j_start:j_end]
          J_W,J_b=self.gradient(Xtr,Ytr,lmd)
          for k in range(len(J_W)):
            self.W[k]=self.W[k]-eta*J_W[k]
            self.b[k]=self.b[k]-eta*J_b[k]
        if shuffle==True:
          tf.random.shuffle(train_X, seed=400)
          tf.random.shuffle(train_Y, seed=400)
          tf.random.shuffle(train_y, seed=400)
      #    tf.random.shuffle(validation_X, seed=400)
      #    tf.random.shuffle(validation_Y, seed=400)
      #    tf.random.shuffle(validation_y, seed=400)
        #if i%10==0:
         # eta_max=eta_max*0.5
         # eta_min=eta_min*0.5
    else:
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
        for j in range(n):
          t=i*n+j
 
          eta=self.generate_eta(eta_min,eta_max,n_s,t)

          j_start=j*batch_size
          j_end=(j+1)*batch_size
          Xtr=train_X[j_start:j_end]
          Ytr=train_Y[j_start:j_end]
          ytr=train_y[j_start:j_end]

          J_W,J_b,J_gamma,J_beta=self.gradient(Xtr,Ytr,lmd)
          for k in range(len(J_W)):
            self.W[k]=self.W[k]-eta*J_W[k]
            self.b[k]=self.b[k]-eta*J_b[k]
          for k in range(len(J_gamma)):
            self.gamma[k]=self.gamma[k]-eta*J_gamma[k]
            self.beta[k]=self.beta[k]-eta*J_beta[k]
        if shuffle==True:
          tf.random.shuffle(train_X, seed=400)
          tf.random.shuffle(train_Y, seed=400)
          tf.random.shuffle(train_y, seed=400)
          tf.random.shuffle(validation_X, seed=400)
          tf.random.shuffle(validation_Y, seed=400)
          tf.random.shuffle(validation_y, seed=400)
    
          
      with open("bn_train_cost.txt","w") as train_cost_file:
        train_cost=str(train_cost)
        train_cost_file.write(train_cost)
      with open("bn_validation_cost.txt","w") as validation_cost_file:
        validation_cost=str(validation_cost)
        validation_cost_file.write(validation_cost)
      with open("bn_train_accuracy.txt","w") as train_accuracy_file:
        train_accuracy=str(train_accuracy)
        train_accuracy_file.write(train_accuracy)
      with open ("bn_validation_accuracy.txt","w") as validation_accuracy_file:
        validation_accuracy=str(validation_accuracy)
        validation_accuracy_file.write(validation_accuracy)


  def ComputeGradsNum(self,X,Y,lamda,h=1e-6):
    grad_W,grad_b,grad_gamma,grad_beta=[],[],[],[]
    _,cost_init=self.computeCost(X,Y,lamda)
    for i in range(len(self.W)):
      grad_W_i=np.zeros([self.W[i].shape[0],self.W[i].shape[1]])
      for j in range(self.W[i].shape[0]):
        for k in range(self.W[i].shape[1]):
          self.W[i][j][k]=self.W[i][j][k]+h
          _,cost_2_W=self.computeCost(X,Y,lamda)
          grad_W_i[j][k]=(cost_2_W-cost_init)/h
          self.W[i][j][k]=self.W[i][j][k]-h
      grad_W.append(grad_W_i)

    for i in range(len(self.b)):
      grad_b_i=np.zeros([self.b[i].shape[0],1])
      for j in range(self.b[i].shape[0]):
        self.b[i][j]=self.b[i][j]+h
        _,cost_2_b=self.computeCost(X,Y,lamda)
        grad_b_i[j]=(cost_2_b-cost_init)/h
        self.b[i][j]=self.b[i][j]-h
      grad_b.append(grad_b_i)

    for i in range(len(self.gamma)):
      grad_gamma_i=np.zeros([self.gamma[i].shape[0],1])
      for j in range(self.gamma[i].shape[0]):
        self.gamma[i][j]=self.gamma[i][j]+h
        _,cost_2_gamma=self.computeCost(X,Y,lamda)
        grad_gamma_i[j]=(cost_2_gamma-cost_init)/h
        self.gamma[i][j]=self.gamma[i][j]-h
      grad_gamma.append(grad_gamma_i)
      

    for i in range(len(self.beta)):
      grad_beta_i=np.zeros([self.beta[i].shape[0],self.beta[i].shape[1]])
      for j in range(self.beta[i].shape[0]):
        self.beta[i][j]=self.beta[i][j]+h
        _,cost_2_beta=self.computeCost(X,Y,lamda)
        grad_beta_i[j]=(cost_2_beta-cost_init)/h
        self.beta[i][j]=self.beta[i][j]-h
      grad_beta.append(grad_beta_i)
      
    return grad_W,grad_b,grad_gamma,grad_beta


  def minibatch(self,train_X,train_Y,train_y,n_batch,eta,lmd,shuffle):
    if shuffle==False:
      n=int(train_X.shape[0]/n_batch)
      for i in range(n):
        j_start=i*n_batch
        j_end=(i+1)*n_batch
        Xtr=train_X[j_start:j_end]
        Ytr=train_Y[j_start:j_end]
        ytr=train_y[j_start:j_end]

        J_W,J_b=self.gradient(Xtr,Ytr,lmd)
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

        J_W,J_b=self.gradient(Xtr,Ytr,lmd)
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

classifier=Classifier(train_X,train_Y,train_y,k=3,batch_noramalization=True)
classifier.addDenseLayer(shape=[50,3072],activation="relu")
classifier.addDenseLayer(shape=[50,50],activation="relu")
classifier.addDenseLayer(shape=[10,50],activation="softmax")

import math
eta_max=1e-1
eta_min=1e-5
batch_size=100

n_s=2250
cycles=2
iterations=2*n_s*cycles
epochs=int(iterations/batch_size)

classifier.cyclic_Minibatch_gd(train_X,train_Y,train_y,epochs=epochs,batch_size=batch_size,eta_min=eta_min,eta_max=eta_max,n_s=n_s,lmd=0.01,shuffle=True)

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

