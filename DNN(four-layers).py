# -*- coding:utf-8 -*-
#搭建一个4层的DNN来分类mnist手写数据集

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#load the data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#define the batch size
batch_size = 100
#define the batch numbers in training set
n_batch = mnist.train.num_examples // batch_size

#define the pleceholder
x = tf.placeholder(tf.float32, [None,784])
y = tf.placeholder(tf.float32, [None,10])
keep_prob = tf.placeholder(tf.float32)
#创建一个简单的神经网络
#输入层为784个神经元，输出层为10个神经元（10个分类），一个隐藏层有20个神经元

#定义第一个中间层
W_L1 = tf.Variable(tf.truncated_normal([784,2000],stddev=0.1)) #一般这样初始化权值
bias_L1 = tf.Variable(tf.zeros([1,2000])+0.1) #一般这样初始化偏置值
signal1 = tf.matmul(x,W_L1) + bias_L1
L1 = tf.nn.tanh(signal1) #为什么第一层的激活函数用tanh？
L1_drop = tf.nn.dropout(L1,keep_prob)

#定义第二个中间层
W_L2 = tf.Variable(tf.truncated_normal([2000,2000],stddev=0.1))
bias_L2 = tf.Variable(tf.zeros([1,2000])+0.1)
signal2 = tf.matmul(L1_drop,W_L2) + bias_L2
L2 = tf.nn.tanh(signal2)
L2_drop = tf.nn.dropout(L2,keep_prob)

#定义第三个中间层
W_L3 = tf.Variable(tf.truncated_normal([2000,1000],stddev=0.1))
bias_L3 = tf.Variable(tf.zeros([1,1000])+0.1)
signal3 = tf.matmul(L2_drop,W_L3) + bias_L3
L3 = tf.nn.tanh(signal3)
L3_drop = tf.nn.dropout(L3,keep_prob)

#定义输出层
W_L4 = tf.Variable(tf.truncated_normal([1000,10],stddev=0.1))
bias_L4 = tf.Variable(tf.zeros([1,10])+0.1)
signal4 = tf.matmul(L3_drop,W_L4) + bias_L4
prediction = tf.nn.softmax(signal4)


#代价函数+梯度下降
# loss_square = tf.reduce_mean(tf.square(y-prediction))
loss_cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
train_step = tf.train.GradientDescentOptimizer(0.7).minimize(loss_cross_entropy)

init = tf.global_variables_initializer()

#定义准确率
#返回一个bool型数组[T,F,F,F,T.....] (T/F表示是否相等)
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1)) #argmax:返回一维张量y中与1最接近的值所在的位置
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32)) #将bool类型转换为f32，即T=1.0，F=0

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(20): #迭代多少个周期
        for batch in range(n_batch): #按批次的数量，一批批迭代
            batch_xs,batch_ys = mnist.train.next_batch(batch_size) #获得一个大小为batch_size=100的批次，传进去，图片数据保存在batch_xs,标签保存在batch_ys，依次运行1-100图片，101-200图片...
            #训练集里所有图片都按批次训练了一遍
            #在for里，获得一个batch，run一次，再获得下一个batch，再run，直到训练完所有batch
            sess.run(train_step,{x:batch_xs,y:batch_ys,keep_prob:0.7})
        #训练完一个周期，看看准确率
        test_acc = sess.run(accuracy,{x:mnist.test.images,y:mnist.test.labels,keep_prob:1.0})
        train_acc = sess.run(accuracy,{x:mnist.train.images,y:mnist.train.labels,keep_prob:1.0})
        print ("Iter"+str(epoch)+"-test accuracy:"+str(test_acc)+"train accuracy:"+str(train_acc))


#accuracy in 20 epochs:
# Iter0-test accuracy:0.9203
# Iter1-test accuracy:0.9304
# Iter2-test accuracy:0.9417
# Iter3-test accuracy:0.9472
# Iter4-test accuracy:0.9447
# Iter5-test accuracy:0.9533
# Iter6-test accuracy:0.9559
# Iter7-test accuracy:0.9561
# Iter8-test accuracy:0.9569
# Iter9-test accuracy:0.9591
# Iter10-test accuracy:0.9629
# Iter11-test accuracy:0.9597
# Iter12-test accuracy:0.9628
# Iter13-test accuracy:0.9614
# Iter14-test accuracy:0.9653
# Iter15-test accuracy:0.9653
# Iter16-test accuracy:0.9648
# Iter17-test accuracy:0.9668
# Iter18-test accuracy:0.968
# Iter19-test accuracy:0.9679