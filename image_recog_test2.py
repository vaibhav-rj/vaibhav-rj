#copyright Vaibhav Raj 2018(email_ID:-vaibhavraj46@gmail.com,vaibhav.raj.phe15@itbhu.ac.in)
#
# Licensed under the Apache License, Version 2.0 (the "License");
#========================================================================================================= 

'''
A simple handwritten devnagari numerals image classifier test code.The purpose of this code is to take in sample test input and classify it as one of the devnagari numerals (from 0-9).Following are the requirements:-
(1)The saved checkpoint that contains our trained session('./image_train2.ckpt') in same location as 'image_recog_train2.py' and 'image_recog_test2.py'.
(2)A handwritten image of 32*32 pixels:-(i)that could be test set of 'DevanagariHandwrittenCharacterDataset',(ii)or could be drawn in MS-Word with (a)either thickest white pencil stroke on black background,or(b)thickest black pencil stroke on white background.
'''
#import modules
import sys
from PIL import Image,ImageFilter
import numpy as np
import tensorflow as tf

'''
This function takes in the image location and recognizes the image as one of the(0-9)devnagari numerals:-
'''
def img_recog(image_location):
	imag=Image.open(image_location)
	image=imag.convert('L')
	image_data=np.array(image,dtype=np.float32)
	image_data = np.multiply(image_data, 1.0/255.0)
	'''
	The below line is for the cases when input image is drawn in white background with black pencil stroke:-
	Change the below line to a comment if the image of devnagari numerals is made by white pencil strokes on a black canvas:-
	'''
	image_data = np.subtract(1.0, image_data)
	
	#Reshape the 'image_batch' from vector to matrix:- 
	image_batch = image_data.reshape(-1, 32,32,1)
    #Creating palceholder for image_batch:- 
	image_batch_pl=tf.placeholder(dtype=tf.float32,shape=[None,32,32,1],name='image_batch_pl')

	'''
	Creating model:-
	'''
	
	#Creating 1st convolution layer:-
	conv1 = tf.layers.conv2d(inputs=image_batch_pl,filters=32,kernel_size=[5, 5],padding="same",activation=tf.nn.relu,name='conv1')

	#Creating 1st pooling layer:-
	pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2,name='pool1')

	#Creating 2nd convolution layer:-
	conv2 = tf.layers.conv2d(inputs=pool1,filters=64,kernel_size=[5, 5],padding="same",activation=tf.nn.relu,name='conv2')
	
	#Creating 2nd pooling layer:-
	pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2,name='pool2')

	#layer associted with flattening 'pool2' matrix into vector:-
	pool2_flat = tf.reshape(pool2, [-1, 8 * 8 * 64],name='pool2_flat')

	#Creating 1st fully connected layer:-
	dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu,name='dense')

	#Creating dropout layer:-
	dropout = tf.layers.dropout(inputs=dense, rate=0.25, training=False,name='dropout')

	#Creating 2nd fully connected layer:-
	logits = tf.layers.dense(inputs=dropout, units=10,name='logits')
	
	#Applying RELU function to 'logits':-
	cost=tf.nn.softmax(logits, name='cost')
	
	#Predicts the index of the maximum element of the cost vector:-
	predict=tf.argmax(cost,axis=1,name='predict')
	
	#Initializing the variables:-
	init = tf.global_variables_initializer()
	
	#Initializing the saver:-
	saver=tf.train.Saver()
	
	#Initializing the new session,restoring the trained model and and predicting the image among(0-9):-
	with tf.Session() as sess:
		sess.run(init)
		saver.restore(sess,'./image_train2.ckpt')
		print('The input image is recognised as the number:')
		print(predict[0].eval(feed_dict={image_batch_pl:image_batch},session=sess))

def main(image_location):
	#Main function
	#Please enter the image location after writing the name of the test code in the command line
	img_recog(image_location)
	
if __name__ == "__main__":
	main(sys.argv[1])