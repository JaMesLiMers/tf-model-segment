# -*- coding:utf-8 -*-
#本CNN共2个卷积层，2个池化层，2个全连接层，分类mnist手写数据集

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data/',one_hot = True)

batch_size = 100
n_batch = mnist.train.num_examples // batch_size


#初始化权值
def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

#初始化偏置值
def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

#定义每一个卷积层中的卷积操作
def conv2d(x,W):
    # x: input tensor. shape:[batch,in_height,in_width,in_channels] 即batch，图片的长宽，通道数
    # W：filter tensor. shape:[filter_height,filter_width,in_channers,put_channels] 即滤波器的长宽，输入输出通道数
    # strides: 步长。第0和3个值需要是1，strides[1]=x方向的步长， strides[2]=y方向的步长
    # padding: ‘SAME'表示在外圈补0，’VALID'表示不补0
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

#定义池化层
def max_pool_2x2(x):
    # x: input tensor.和上面一样
    # ksize:池化窗口的大小。第0和第3需要是1，[1]和[2]决定长和宽
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

#定义3个placeholder, for 图片和label;以及dropout的概率
x = tf.placeholder(tf.float32,[None,784]) #28*28
y = tf.placeholder(tf.float32,[None,10])
keep_prob = tf.placeholder(tf.float32)


#改变x的格式，转化为4D向量[batch,in_height,in_width,in_channels],以符合卷积和池化的tensor格式
x_image = tf.reshape(x,[-1,28,28,1]) #把一维的1*784的图片转化为二维的tensor格式

#初始化第一个卷积层的权值和偏置值
W_conv1 = weight_variable([5,5,1,32]) #传入shape，5*5的采样窗口，32个卷积核从1个平面中采样
b_conv1 = bias_variable([1,32]) #每1个卷积核设置1个偏置值，共32个卷积核

#第一次卷积操作，此时得到32个特征平面,图片尺寸为28*28
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1) #将第一个卷积层的权值和偏置值传入卷积操作中,x_image作为输入图像tensor，卷积后再用relu激活函数输出，作为第一个卷积层的输出

#第一次池化操作，此时还是32个特征平面，图片尺寸为14*14
h_pool1 = max_pool_2x2(h_conv1) #将第一次卷及操作后的结果，作为输入到第一个池化层进行池化操作

#初始化第二个卷积层的权值和偏置值
W_conv2 = weight_variable([5,5,32,64]) #传入shape，5*5的采样窗口，64个卷积核从32个特征平面中采样
b_conv2 = bias_variable([1,64]) #共64个卷积核

#第二次卷积操作，此时得到64个特征平面，图片尺寸为14*14
h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2) #第一次池化的结果为h_pool1，作为输入传到第二个卷积层

#第二次池化操作，此时还是64个特征平面，图片尺寸为7*7
h_pool2 = max_pool_2x2(h_conv2)

#现在共有64张7*7的特征图

#初始化第一个全连接层
W_fc1 = weight_variable([7*7*64,1024]) #全连接，上一层有7*7*64个像素点（神经元），全连接层1024个神经元
b_fc1 = bias_variable([1,1024])

#把池化层2的输出扁平化为1维，原本为卷积操作的输入，即[100,7,7,64] as [batch,in_height,in_width,in_channels]
h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64]) # -1代表任意值，这里是batch_size，然后把后面3个维度转化为了1个维度，即7*7*64

#求第一个全连接层的输出
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1) + b_fc1)

#求第一个全连接层的输出的同时再加上dropout,让部分神经元不工作
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

#初始化第二个全连接层
W_fc2 = weight_variable([1024,10]) #全连接，上一层有1024个神经元,输出10个神经元
b_fc2 = bias_variable([1,10])

#计算第二个全连接层的输出
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2) + b_fc2)

#设置交叉熵代价函数，因为预测结果是用softmax激活，是非线性，所以不用平方代价函数
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))

#使用AdamOptimizer进行优化
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

#结果存放在一个bool列表中
correction_prediction = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1)) #如果predicrion和y是相同的位置，则该item值置为1

#bool列表的值转化为float32再求平均值
accuracy = tf.reduce_mean(tf.cast(correction_prediction,tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(20):
        for batch in range(n_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step,{x:batch_xs,y:batch_ys,keep_prob:0.7})

        acc = sess.run(accuracy,{x:mnist.test.images,y:mnist.test.labels,keep_prob:1.0})
        print ('Iter'+str(epoch)+':accuracy='+str(acc))


#accuracy in 20 epochs:
# Iter0:accuracy=0.8644
# Iter1:accuracy=0.9687
# Iter2:accuracy=0.9792
# Iter3:accuracy=0.9814
# Iter4:accuracy=0.9834
# Iter5:accuracy=0.9864
# Iter6:accuracy=0.9871
# Iter7:accuracy=0.9874
# Iter8:accuracy=0.9877
# Iter9:accuracy=0.9882
# Iter10:accuracy=0.9881
# Iter11:accuracy=0.9901
# Iter12:accuracy=0.9896
# Iter13:accuracy=0.9896
# Iter14:accuracy=0.9907
# Iter15:accuracy=0.9903
# Iter16:accuracy=0.9913
# Iter17:accuracy=0.9911
# Iter18:accuracy=0.9907
# Iter19:accuracy=0.9915