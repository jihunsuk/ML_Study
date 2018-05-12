import tensorflow as tf

# 매트릭스 사용  
x_data = [[73., 80., 75.], [93., 88., 93.],
          [89., 91., 90.], [96., 98., 100.], [73., 66., 70.]]
y_data = [[152.], [185.], [180.], [196.], [142.]]

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

#매트릭스를 사용함으로 코드가 더 깔끔해졌다