#copyright Vaibhav Raj 2018(email_ID:-vaibhav.raj.phe15@itbhu.ac.in,vaibhavraj46@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
#==========================================================================================================================

'''
A simple handwritten Devnagari numerals(0-9) classifier for training model.For training we would use the extracted version of the zip file :-'DevanagariHandwrittenCharacterDataset' that I've already uploaded.The extracted version needs to be saved along with the 'image_recog_train2.py'(file containing code for training the model) and 'image_recog_test2.py'(file containing code for classifying any random test image).
'''

#import modules
from PIL import Image
import numpy as np
import tensorflow as tf
import os
import re

'''
Algorithm to create two lists:-
(1)'image_batch' list for holding image color intensity matrix of all the handwritten devnagari numerals present in the train set of the extracted version of 'DevanagariHandwrittenCharacterDataset'.
(2)'labels' list for holding respective labels of the elements of 'image_batch'.
'''
L=[]
image_data=[]
labels=[]
path='DevanagariHandwrittenCharacterDataset\\DevanagariHandwrittenCharacterDataset\\Train'
l=os.listdir(path)
D=[]

for i in l:
    if(re.search('digit',i)):
        D.append(i)
        m=os.listdir(path+'\\'+i)
        L.append(m)

for j in range(len(m)):
    for k in D:
        d=int(k[::-1][0])
        n=L[d][j]
        imag=Image.open(path+'\\'+k+'\\'+n)
        image=imag.convert(mode='L')
        image_data.append(np.array(image,dtype=np.float32))
        labels.append(d)

image_data = np.multiply(image_data, 1.0/255.0)

image_batch = image_data.reshape(-1, 32,32,1)
labels=np.asarray(labels, dtype=np.int32)			

#Creating placeholders:-image_batch_pl(for 'image_batch' list) and labels_pl(for 'labels' list):- 
image_batch_pl=tf.placeholder(dtype=tf.float32,shape=[None,32,32,1],name='image_batch_pl')
labels_pl=tf.placeholder(dtype=np.int32,shape=[None])

'''
Creating CNN(convoluted neural network) model:-
'''

#Creating 1st convolution layer:-
conv1 = tf.layers.conv2d(inputs=image_batch_pl,filters=32,kernel_size=[5, 5],padding="same",activation=tf.nn.relu,name='conv1')

#creating 1st pooling layer:-
pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2,name='pool1')

#Creating 2nd convolution layer:-
conv2 = tf.layers.conv2d(inputs=pool1,filters=64,kernel_size=[5, 5],padding="same",activation=tf.nn.relu,name='conv2')
 
#Creating 2nd pooling layer:- 
pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2,name='pool2')

#Flattening the pool2 matrix into vector:-
pool2_flat = tf.reshape(pool2, [-1, 8 * 8 * 64],name='pool2_flat')

#Creating fully connected layer 'dense' with RELU activation:-
dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu,name='dense')

#Creating dropout layer:-
dropout = tf.layers.dropout(inputs=dense, rate=0.25, training=True,name='dropout')

#Again creatinf a fully connected layer:-
logits = tf.layers.dense(inputs=dropout, units=10,name='logits')

#Applying softmax activation function to 'logits':-
cost=tf.nn.softmax(logits, name='cost')

'''
Model created:-
'''
#Converting the labels into one-hot vector:-
labelV=tf.one_hot(labels_pl,depth=10,name='labelV')

'''
Defining the loss function,optimizer,saver(for saving and restoring the session),batch size,iterations(as iters here),epochs(here its 30) along with initializing a new session and variables:-
'''
loss = -tf.reduce_sum(labelV*tf.log(cost),name='loss')
sess=tf.Session()
init=tf.global_variables_initializer()
sess.run(init)
train=tf.train.GradientDescentOptimizer(0.001).minimize(loss)
saver=tf.train.Saver()
batch_size=50
iter=(len(labels)//50)

'''
Training the model:-
'''
for h in range(20):
	for k in range(iter):
		sess.run(train,feed_dict={image_batch_pl:image_batch[batch_size*k:batch_size*(k+1)],labels_pl:labels[batch_size*k:batch_size*(k+1)]})
		
	print(("{0} epochs").format(h))
		
'''
Saving the session as a checkpoint file:-'./image_train2.ckpt' and printing the confirmation message:-
'''		
save_path = saver.save(sess, './image_train2.ckpt')
print ("Model saved in file: ", save_path)