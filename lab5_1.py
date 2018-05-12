# Logistic Regression     0인지 1인지를 분류하는것. ex) 스팸메일인지 아닌지 등등
# 이전에 배운 linear regression처럼 H(x)를 같게하면 최소점을 잘 못찾는다. 그래서 식이 조금 변경된다. 변경되는 식이 sigmoid로 표현된다.
# sigmoid 값은 0보다크고 1보다 작은값이 나온다.
import tensorflow as tf
x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]

X = tf.placeholder(tf.float32, shape=[None, 2])
Y = tf.placeholder(tf.float32, shape=[None, 1])
W = tf.Variable(tf.random_normal([2, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# 0.5보다 크면 true 작으면 false  이를 float32로 cast하면 true면 1 false면 0으로 나온다.
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
# Y랑 predicted를 비교한다. true면 1  false 0  이를 reduce_mean을 통해 얼마나 정확한지 측정한다.
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #학습 시작
    for step in range(10001):
        cost_val, _ = sess.run([cost, train], feed_dict={X: x_data, Y: y_data})
        if step % 200 == 0:
            print(step, cost_val)

    # 학습이 끝나고 정확성 보고
    h, c, a = sess.run([hypothesis, predicted, accuracy],
                       feed_dict={X: x_data, Y: y_data})
    print("\nHypothesis: ", h, "\nCorrect (Y): ", c, "\nAccuracy: ", a)