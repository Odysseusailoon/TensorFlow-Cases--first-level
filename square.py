# TensorFlow-Cases--first-level
##This is the first tensorflow case I've created.
import tensorflow as tf
import numpy as np
##create data
x_data = np.random.randn(100).astype(np.float32)
y_data = x_data*0.1 +0.3
##create tensorflow structure start!
Weight = tf.Variable(tf.random_uniform([1],-1.0,1.0))#creat a variable by random_uniform, [1]means it's one demension,-1.0~1.0it's the distance
biases = tf.Variable(tf.zeros([1]))#we ser biases as 0 at the first time by using tf.zeros
#set a function!
y = Weight*x_data + biases
# set the loss function by using the square
loss = tf.reduce_mean(tf.square(y-y_data))
#set the opitimizer by using GradientDescent! the step is 0.5.
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)
#initilize variables~~
init = tf.initialize_all_variables()
#cyclic
sess = tf.Session()
sess.run(init)

for step in range(201):
    sess.run(train)
    if step%20 == 0:
        print(step, sess.run(Weight), sess.run(biases))
