# 간단한 linear regression구현

import tensorflow as tf

#텐서플로우가 사용할 변수선언
W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

#input
X = tf.placeholder(tf.float32, shape=[None])
Y = tf.placeholder(tf.float32, shape=[None])

#평균값을 얻어낸다.
hypothesis = X * W + b
cost = tf.reduce_mean(tf.square(hypothesis - Y))

#cost가 0이되게 최적화한다. Gradient Descent 알고리즘을 이용해 조금씩 바꾸면서 cost가 낮게만든다.
#기울기가 양이면 공식에따라 W가 줄어들고 기울기가 음이염 W가 늘어난다.
#이 공식을 쓰기위해서는 cost가 convex function가 되어야한다.
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer()) #텐서플로우가 사용할 변수 초기화해줘야된다.

# _ 는 반환안되게 해준다.
for step in range(2001):
    cost_val, W_val, b_val, _ = sess.run([cost, W, b, train],
                                         feed_dict={X:[1,2,3,4,5],
                                                    Y: [2.1, 3.1, 4.1, 5.1, 6.1]})
    if step % 20 == 0:  #너무 많으니 20번마다 출력
        print(step, cost_val, W_val, b_val)

#결과적으로 W는 1, b는 1.1에 가까워진다.