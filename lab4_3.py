# 데이터들을 직접쓰면 힘드니까 파일에서 불러오는 방법이다.
# .csv 라는 파일로부터 읽어온다.
import tensorflow as tf
import numpy as np

xy = np.loadtxt('lab4_3.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]   #행은 전체 렬은 0부터 -1(마지막까지) 이므로 처음부터 마지막을 제외하고 다가져옴
y_data = xy[:, [-1]]   #행은 전체 렬을 마지막꺼 가져옴

print(x_data.shape, x_data, len(x_data))
print(y_data.shape, y_data)

X = tf.placeholder(tf.float32, shape=[None, 3])   #n x 3 Matrix  위에는 5개라서 원래 5 x 1이지만 n개넣을수있다.
Y = tf.placeholder(tf.float32, shape=[None, 1])   #n x 1 Matrix

W = tf.Variable(tf.random_normal([3, 1]), name='weight1')
b = tf.Variable(tf.random_normal([1]), name='bias')
hypothesis =tf.matmul(X, W) + b   #matmul은 매트릭스 곱셈연산

cost = tf.reduce_mean(tf.square(hypothesis - Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
                                   feed_dict={X: x_data, Y:y_data})
    if step % 10 == 0:
        print(step, "cost: ", cost_val, "\nPrediction:\n", hy_val)

#이제 학습했으니 물어본다
print("Your score will be ", sess.run(hypothesis, feed_dict={X: [[100, 70, 101]]}))