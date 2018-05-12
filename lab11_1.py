
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

sess = tf.InteractiveSession()
image = np.array([[[[1],[2],[3]],   #이런 이미지가 있다고 하자
                   [[4],[5],[6]],
                   [[7],[8],[9]]]], dtype=np.float32)
print(image.shape)
plt.imshow(image.reshape(3,3), cmap='Greys')

print("image.shape", image.shape)
weight = tf.constant([[[[1.]],[[1.]]],
                      [[[1.]],[[1.]]]])
print("weight.shape", weight.shape)
conv2d = tf.nn.conv2d(image, weight, strides=[1, 1, 1, 1], padding='VALID') #image를 strides의 2번째와 3번째값인 1x1로 샘플링한다.
conv2d_img = conv2d.eval()
print("conv2d_img.shape", conv2d_img.shape)
conv2d_img = np.swapaxes(conv2d_img, 0, 3)
for i, one_img in enumerate(conv2d_img):
    print(one_img.reshape(2,2))
    plt.subplot(1,2,i+1), plt.imshow(one_img.reshape(2,2), cmap='gray')

#1+2+4+5    2+3+5+6
#4+5+7+8    5+6+8+9 가 된다.

print("image.shape", image.shape)

weight = tf.constant([[[[1.]],[[1.]]],
                      [[[1.]],[[1.]]]])
print("weight.shape", weight.shape)
conv2d = tf.nn.conv2d(image, weight, strides=[1, 1, 1, 1], padding='SAME')  #패딩을 same으로하면 필러되서 나오는것도 같이 3x3이 된다.
conv2d_img = conv2d.eval()
print("conv2d_img.shape", conv2d_img.shape)
conv2d_img = np.swapaxes(conv2d_img, 0, 3)
for i, one_img in enumerate(conv2d_img):
    print(one_img.reshape(3,3))
    plt.subplot(1,2,i+1), plt.imshow(one_img.reshape(3,3), cmap='gray')

# 이번에는 여러개의 필터 사용
print("image.shape", image.shape)

weight = tf.constant([[[[1., 10., -1.]], [[1., 10., -1.]]], #필터를 3으로 늘린다. 그러므로 3장의 이미지가 나온다.
                      [[[1., 10., -1.]], [[1., 10., -1.]]]])
print("weight.shape", weight.shape)
conv2d = tf.nn.conv2d(image, weight, strides=[1, 1, 1, 1], padding='SAME')  #이렇게 하나만해도 필터를 3으로해서 3개가 나온다.
conv2d_img = conv2d.eval()
print("conv2d_img.shape", conv2d_img.shape)
conv2d_img = np.swapaxes(conv2d_img, 0, 3)
for i, one_img in enumerate(conv2d_img):
    print(one_img.reshape(3, 3))
    plt.subplot(1, 3, i + 1), plt.imshow(one_img.reshape(3, 3), cmap='gray')


# 이제 풀링을 하면된다.
image = np.array([[[[4],[3]],
                    [[2],[1]]]], dtype=np.float32)
pool = tf.nn.max_pool(image, ksize=[1, 2, 2, 1],       # max pooling을 많이사용한다.
                    strides=[1, 1, 1, 1], padding='VALID')
print(pool.shape)
print(pool.eval())

image = np.array([[[[4],[3]],
                    [[2],[1]]]], dtype=np.float32)
pool = tf.nn.max_pool(image, ksize=[1, 2, 2, 1],
                    strides=[1, 1, 1, 1], padding='SAME')
print(pool.shape)
print(pool.eval())

# 이제 실제 이미지를 해보자.
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

img = mnist.train.images[0].reshape(28,28)  # 데이터의 가장 첫번째 이미지 해보자.
plt.imshow(img, cmap='gray')

sess = tf.InteractiveSession()

img = img.reshape(-1,28,28,1)   #28x28의 1color의 이미지다 -1은 n개의 이미지중 몇개라는 의미인곳인데 -1하면 알아서 계산해라는 의미
W1 = tf.Variable(tf.random_normal([3, 3, 1, 5], stddev=0.01))   #3x3에 1color에 5개의 필터
conv2d = tf.nn.conv2d(img, W1, strides=[1, 2, 2, 1], padding='SAME')    #stride가 2이므로 14x14가 나온다.
print(conv2d)
sess.run(tf.global_variables_initializer())
conv2d_img = conv2d.eval()
conv2d_img = np.swapaxes(conv2d_img, 0, 3)
for i, one_img in enumerate(conv2d_img):
    plt.subplot(1,5,i+1), plt.imshow(one_img.reshape(14,14), cmap='gray')   #출력하면 필터가 5니까 5개의 이미지가 나온다.


pool = tf.nn.max_pool(conv2d, ksize=[1, 2, 2, 1], strides=[ # 14x14에서 7x7로 max pool된다.
                        1, 2, 2, 1], padding='SAME')
print(pool)
sess.run(tf.global_variables_initializer())
pool_img = pool.eval()
pool_img = np.swapaxes(pool_img, 0, 3)
for i, one_img in enumerate(pool_img):
    plt.subplot(1,5,i+1), plt.imshow(one_img.reshape(7, 7), cmap='gray')