import tensorflow as tf
import numpy as np
tf.set_random_seed(777)  # for reproducibility

# Predicting animal type based on various features
xy = np.loadtxt('lab6_2.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

print(x_data.shape, y_data.shape)

nb_classes = 7  # 0 ~ 6   7개로 분류된다.

X = tf.placeholder(tf.float32, [None, 16])  #16개의 x값이 있다.
Y = tf.placeholder(tf.int32, [None, 1])  # 0 ~ 6
Y_one_hot = tf.one_hot(Y, nb_classes)  # 데이터가 0~6으로 one hot값이 아니기 때문에 one hot으로 바꿔줘야한다.  원핫으로 만들면 1차원 더 크게만든다. rank가 N이라면 출력이 N+1이 된다. [[0], [3]]이면 one hot인경우
                                                                                                            #rank가 7이니까 [[[ 1000000]], [[0001000]]] 이렇게된다.  0이면첫번째 3이면 3번째가 hot이 되니까.  1차원이 늘어난다. 그래서 줄여야한다.
print("one_hot", Y_one_hot)
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])   #그래서 다시 1차원을 줄이기위해 reshape한다.
print("reshape", Y_one_hot)

W = tf.Variable(tf.random_normal([16, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

# tf.nn.softmax computes softmax activations
# softmax = exp(logits) / reduce_sum(exp(logits), dim)
logits = tf.matmul(X, W) + b
hypothesis = tf.nn.softmax(logits)

# Cross entropy cost/loss   6_1에서의 식을 간단하게 하기위해 텐서플로우에서 지원하는 함수가 softmax_cross_entropy_with_logits이다.
cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits,      #logit은 tf.matmul(X, W) + b
                                                 labels=Y_one_hot)   #label은 Y값
cost = tf.reduce_mean(cost_i)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

prediction = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# Launch graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2000):
        sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
        if step % 100 == 0:
            loss, acc = sess.run([cost, accuracy], feed_dict={
                                 X: x_data, Y: y_data})
            print("Step: {:5}\tLoss: {:.3f}\tAcc: {:.2%}".format(
                step, loss, acc))

    # Let's see if we can predict
    pred = sess.run(prediction, feed_dict={X: x_data})
    # y_data: (N,1) = flatten => (N, ) matches pred.shape
    for p, y in zip(pred, y_data.flatten()):    # flatten은 [[1], [0]] 을 [1, 0] 이런식으로 평평하게 해주는것
        print("[{}] Prediction: {} True Y: {}".format(p == int(y), p, int(y)))