# Patel, Nabilahmed
# 1001-234-817
# 2016-11-12
# Assignment_05

import theano
from theano import tensor as T
import numpy as np
import scipy.misc
import matplotlib
from matplotlib import pyplot as plt
import os
import sys
from os import listdir
from os.path import isfile, join

def model(train_input,train_target,test_input,test_target,no_of_nodes,alpha,_lambda,activation):

    X=T.fmatrix('input')
    Y=T.fmatrix('target')

    #weights of hidden layer nodes
    w_h = np.random.uniform(-0.1, 0.1,(3072,no_of_nodes))               #no_of_nodes = 100
    W_H = theano.shared(np.asarray(w_h, dtype=theano.config.floatX))
    #bais of hidden layer nodes
    b_h = np.random.uniform(-0.1, 0.1,(1,no_of_nodes))
    B_H = theano.shared(np.asarray(b_h, dtype=theano.config.floatX),broadcastable=[True,False])
    #weights of output layer nodes
    w_o = np.random.uniform(-0.1, 0.1,(no_of_nodes,10))
    W_O = theano.shared(np.asarray(w_o, dtype=theano.config.floatX))
    #bais of output layer nodes
    b_o = np.random.uniform(-0.1, 0.1,(1,10))
    B_O = theano.shared(np.asarray(b_o, dtype=theano.config.floatX),broadcastable=[True,False])

    #calculating output from both layer
    if activation == "relu":
        h_output = T.nnet.relu((T.dot(X,W_H)) + B_H)
    elif activation == "sigmoid":
        h_output = T.nnet.sigmoid((T.dot(X,W_H)) + B_H)
    o_output = T.nnet.softmax((T.dot(h_output,W_O))+ B_O)

    #finding maximum value node of each sample for whole data
    _max = T.argmax(o_output, axis=1)
    
    #calculating loss
    cost = T.mean(T.nnet.categorical_crossentropy(o_output, Y))

    #calculating gradient
    dWH, dBH, dWO, dBO = T.grad(cost,wrt=[W_H, B_H, W_O, B_O])

    #updating weights
    _update = [[W_H,W_H - (alpha * dWH) + ( _lambda * T.square(W_H)) ],[B_H,B_H - (alpha * dBH)],[W_O,W_O - (alpha * dWO) + (_lambda * T.square(W_O))],[B_O,B_O - (alpha * dBO)]]

    #training function
    train = theano.function(inputs=[X,Y], outputs=cost, updates=_update, allow_input_downcast=True)
    #error finding funtion
    predict = theano.function(inputs=[X], outputs=_max, allow_input_downcast=True)

    #training
    _loss = []
    Err_rate = []
    for epoch in range(200):
        loss = []
        print(epoch)
        for sample in range(1000):
            #loss for each sample
            cost = train(train_input[:,sample].reshape(1,-1),train_target[:,sample].reshape(1,-1))
            loss.append(cost)
        _loss.append(np.mean(np.array(loss)))
        #calculate Error Rate
        Err_rate.append(np.mean(np.argmax(train_target.transpose(), axis=1) == predict(train_input.transpose())))
    
    #testing
    test_max = predict(test_input.transpose())
    #confusion matrix
    CM = np.array([np.zeros(10) for i in range(0,10)])
    for row,col in zip(test_max,np.argmax(test_target.transpose(),axis=1)):
            CM[row][col] = CM[row][col]+1
            
    return [_loss,Err_rate,CM]

def reading_input(path):

    #reading and making input
   
    i = 1
    files = []

    for filename in os.listdir(path):
        if isfile(join(path,filename)):
            files.append(filename)
            files = np.array(files)
            #shuffling files
            np.random.shuffle(files)
            files  = files.tolist()                                   
                                    
    for filename in files:
        img = scipy.misc.imread(join(path,filename)).astype(np.float32)  #read image and convert to float
        img = img.reshape(-1,1)  #reshape to column vector 
        t = np.zeros(10)
        t[int(filename[0])] = 1
        t = t.reshape(-1,1)
        if i == 1:
            P = np.array(img)
            
            targets = np.array(t)            
            i = i + 1
        else:	
            #adding each image to final input 			
            P = np.concatenate((P,img),axis=1)
            #adding each targets to final main targets by changing dimention i.e. from [1 X 10] to [10 X 1]				
            targets = np.concatenate((targets,t),axis=1)

    P = np.divide(P,255)
    return [P,targets]

def display(_loss,Err_rate,CM,colors,labels):
    
    epoch = [i for i in range(0,200)]
    
    #plotting loss
    for l,color,lab in zip(_loss,colors,labels):
        fig1=plt.figure('Loss')
        ax1=fig1.add_subplot(111)
        ax1.plot(epoch, l, color, label=lab)
        plt.legend()
    
    #plotting error_rate
    for e,color,lab in zip(Err_rate,colors,labels):
        fig2=plt.figure('Error Rate')
        ax2=fig2.add_subplot(111)
        ax2.plot(epoch, e, color, label=lab)
        plt.legend()
    plt.show()

    #printing Confusion Matrix
    print(CM)


#command line arguments
while(True):
    task_no = raw_input('please input the task no(1 to 5) enter any other number to quit: ')

	#reading training and testing data
    
    if task_no == '5':
        train_path = "cifar_data_1000_100/train/"
        test_path = "cifar_data_1000_100/test/"
    elif task_no in ['1','2','3','4']:
        train_path = "cifar_data_100_10/train/" 
        test_path = "cifar_data_100_10/test/"
    else:
        sys.exit()
        
    train_data = reading_input(train_path)
    train_input = train_data[0]
    train_target = train_data[1]
    test_data = reading_input(test_path)
    test_input = test_data[0]
    test_target = test_data[1]


    #model(train_input,train_target,test_input,test_target,no_of_nodes,alpha,_lambda,activation)
    if task_no == '1':
        output = model(train_input,train_target,test_input,test_target,100,0.005,0.0,"relu")
        display([output[0]],[output[1]],[output[2]],['b-'],["Task1"])
    elif task_no == '2':
        output = model(train_input,train_target,test_input,test_target,100,0.005,0.0,"sigmoid")
        display([output[0]],[output[1]],[output[2]],['b-'],["Task2"])
    elif task_no == '3':
        output1 = model(train_input,train_target,test_input,test_target,100,0.005,0.0,"relu")
        output2 = model(train_input,train_target,test_input,test_target,200,0.005,0.0,"relu")
        output3 = model(train_input,train_target,test_input,test_target,300,0.005,0.0,"relu")
        output4 = model(train_input,train_target,test_input,test_target,400,0.005,0.0,"relu")
        output5 = model(train_input,train_target,test_input,test_target,500,0.005,0.0,"relu")
        display([output1[0],output2[0],output3[0],output4[0],output5[0]],[output1[1],output2[1],output3[1],output4[1],output5[1]],[output1[2],output2[2],output3[2],output4[2],output5[2]],['b-','r-','c-','g-','k-'],["No._of_nodes-100","200","300","400","500"])
    elif task_no == '4':
        output1 = model(train_input,train_target,test_input,test_target,500,0.005,0.1,"relu")
        output2 = model(train_input,train_target,test_input,test_target,500,0.005,0.2,"relu")
        output3 = model(train_input,train_target,test_input,test_target,500,0.005,0.3,"relu")
        output4 = model(train_input,train_target,test_input,test_target,500,0.005,0.4,"relu")
        output5 = model(train_input,train_target,test_input,test_target,500,0.005,0.5,"relu")
        display([output1[0],output2[0],output3[0],output4[0],output5[0]],[output1[1],output2[1],output3[1],output4[1],output5[1]],[output1[2],output2[2],output3[2],output4[2],output5[2]],['b-','r-','c-','g-','k-'],["Lambda-0.1","0.2","0.3","0.4","0.5"])
    elif task_no == '5':
        output = model(train_input,train_target,test_input,test_target,300,0.008,0.09,"sigmoid")
        display([output[0]],[output[1]],[output[2]],['b-'],["Task5"])
    else:
        sys.exit()
