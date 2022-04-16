%\load data
function [X,Y,y]=LoadBatch(path)
    A=load(path);
    for i=1:size(A.labels,1)
        A.labels(i)=A.labels(i)+1;
    end
    labels=categorical(A.labels);
    Y=onehotencode(labels,2)';
    X=double(A.data');
    y=double(A.labels');
end
%set hyper-parameters 
function Params=setParams(n_batch,eta,n_epochs)
    Params.n_batch=n_batch;
    Params.eta=eta;
    Params.n_epochs=n_epochs;
end
%rearrange the range of X
function [mean_X,std_X,X]=PreProcess(X)
    mean_X=mean(X,2);
    std_X=std(X,0,2);
    X = X-repmat(mean_X,[1,size(X,2)]);
    X = X ./ repmat(std_X, [1,size(X,2)]);
end
%initialize network parameters W b
function [W,b]=ParaInit(X,Y)
    mean=0;
    dev=0.01;
    %K should be Kcategories*num of dimensions one example possesses 
    %dimb should be K*1
    K=size(Y,1);
    d=size(X,1);
    W=mean+dev*randn(K,d);
    b=mean+dev*randn(K,1);
end
%output the weight sum of the network
function P= EvaluateClassifier(X, W, b)
    b=repmat(b,1,size(X,2));
    s=W*X+b;
    temp=repmat(sum(exp(s),1),size(W,1),1);
    P=exp(s)./temp;
end
%compute the accuracy

function acc=ComputeAccuracy(X,y,W,b)
    P=EvaluateClassifier(X,W,b);
    [~,position]=max(P);
    acc=length(find(position==y))/length(y);
end
%compute the cost
function J = ComputeCost(X, Y, W, b, lambda)
    P = EvaluateClassifier(X, W, b);
    J1 = trace(-log(Y'*P))/size(X, 2);
    J2 = lambda*sum(sum(W.^2));
    J = J1 + J2;
end

%compute the gradient
function [grad_W, grad_b] = ComputeGradients(train_X, train_Y,P,W,lambda)

no = size(W, 1);
grad_W = zeros(size(W));
grad_b = zeros(no, 1);

g= -(train_Y - P);
grad_W=grad_W+g*train_X'/size(train_X,2)+2*lambda*W;
grad_b=(grad_b+g*ones(size(train_X,2),1))./size(train_X,2);
end
%mini batch training
function [Wstar,bstar]=MiniBatchGD(X,Y,n_batch,eta,W,b,lambda)
    for j=1:size(X,2)/n_batch
        j_start=(j-1)*n_batch+1;
        j_end=j*n_batch;
        %inds=j_start:j_end;
        Xbatch=X(:,j_start:j_end);
        Ybatch=Y(:,j_start:j_end);
        P=EvaluateClassifier(Xbatch,W,b);
        [grad_W, grad_b] = ComputeGradients(Xbatch,Ybatch, P, W, lambda);
        W=W-eta*grad_W;
        b=b-eta*grad_b;
    end
    Wstar=W;
    bstar=b;
end
%the gradient comparison
clear
addpath '\\ug.kth.se\dfs\home\y\U\yuchenga\appdata\xp.V2\Desktop\DD2424\Datasets\cifar-10-batches-mat\'
path1='data_batch_1.mat';
lambda=0;
n_batch=200;
eta=0.001;
n_epochs=100;
train_inds=1;20;
test_inds=1000:1500;
acc_train=zeros(1,n_epochs);
acc_test=zeros(1,n_epochs);
J_train=zeros(1,n_epochs);
J_test=zeros(1,n_epochs);
[X,Y,y]=LoadBatch(path1);
[mean_X,std_X,X]=PreProcess(X);
train_X=X(:,train_inds);
train_Y=Y(:,train_inds);
train_y=y(:,train_inds);
[W,b]=ParaInit(train_X,train_Y);
P= EvaluateClassifier(train_X, W, b);
[n_grad_b,n_grad_W]=ComputeGradsNumSlow(train_X(:,1),train_Y(:,1),W,b,lambda,1e-6);
[grad_W, grad_b] = ComputeGradients(train_X, train_Y, P, W, lambda);
errW=sum(sum(abs(n_grad_W-grad_W)./max(eps,abs(n_grad_W)+abs(grad_W))))./numel(grad_W);
errb=sum(sum(abs(n_grad_b-grad_b)./max(eps,abs(n_grad_b)+abs(grad_b))))./numel(grad_b);

%the main function
clear

addpath '\\ug.kth.se\dfs\home\y\U\yuchenga\appdata\xp.V2\Desktop\DD2424\Datasets\cifar-10-batches-mat\'
path={'data_batch_1.mat','data_batch_2.mat','data_batch_3.mat','data_batch_4.mat','data_batch_5.mat'};
path_test='test_batch.mat';
Xtr=[];Ytr=[];ytr=[];
%loading data
for i=1:5
   [X_temp,Y_temp,y_temp]=LoadBatch(path{i});
   Xtr=[Xtr X_temp];
   Ytr=[Ytr Y_temp];
   ytr=[ytr y_temp];
end
Xva=Xtr;Yva=Ytr;yva=ytr;
%load test
[Xte,Yte,yte]=LoadBatch(path_test);
%hyper-parameters
lambda=0;
n_batch=100;
eta=0.001;
n_epochs=40;
train_inds=1:20000;
validation_inds=1:10000;
test_inds=1:10000;
%pre-allocate space

J_tr=zeros(1,n_epochs);
J_va=zeros(1,n_epochs);
J_te=zeros(1,n_epochs);
%preprocess
[mean_Xtr,std_Xtr,Xtr]=PreProcess(Xtr);
[mean_Xva,std_Xva,Xva]=PreProcess(Xva);
[mean_Xte,std_Xte,Xte]=PreProcess(Xte);

train_X=Xtr(:,train_inds);
train_Y=Ytr(:,train_inds);
train_y=ytr(:,train_inds);
validation_X=Xva(:,validation_inds);
validation_Y=Yva(:,validation_inds);
validation_y=yva(:,validation_inds);
test_X=Xva(:,test_inds);
test_Y=Yva(:,test_inds);
test_y=yva(:,test_inds);
%parameter initialize
[Wstar,bstar]=ParaInit(train_X,train_Y);

for i=1:n_epochs
    rng(400);shuffle=randperm(size(train_X,2));
    train_X=train_X(:,shuffle);train_Y=train_Y(:,shuffle);train_y=train_y(:,shuffle);
    J_tr(i)=ComputeCost(train_X, train_Y,Wstar,bstar,lambda);    
    J_va(i)=ComputeCost(validation_X,validation_Y,Wstar,bstar,lambda);
    [Wstar,bstar]=MiniBatchGD(train_X,train_Y,n_batch,eta,Wstar,bstar,lambda);
end
acc_tr=ComputeAccuracy(train_X,train_y,Wstar,bstar);
acc_te=ComputeAccuracy(test_X,test_y,Wstar,bstar);
l_tr=J_tr-lambda*sumsqr(Wstar);
l_va=J_va-lambda*sumsqr(Wstar);
inds=1:n_epochs;
figure()
plot(inds,J_tr,'r');
hold on
plot(inds,J_va,'b');
hold off
xlabel('epochs');
ylabel('cost');
legend('train cost','validation cost');
inds=1:n_epochs;
figure()
plot(inds,l_tr,'r');
hold on
plot(inds,l_va,'b');
hold off
xlabel('epochs');
ylabel('loss');
legend('train loss','validation loss');
for i=1:10
 im = reshape(Wstar(i, :), 32, 32, 3);
 s_im{i} = (im - min(im(:))) / (max(im(:)) - min(im(:)));
 s_im{i} = permute(s_im{i}, [2, 1, 3]);
end
%montage(s_im)


