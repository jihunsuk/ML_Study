import tensorflow as tf
import numpy as np

tf.set_random_seed(777)  # for reproducibility
learning_rate = 0.01

x_data = [[0, 0],
          [0, 1],
          [1, 0],
          [1, 1]]
y_data = [[0],
          [1],
          [1],
          [0]]
x_data = np.array(x_data, dtype=np.float32)
y_data = np.array(y_data, dtype=np.float32)

X = tf.placeholder(tf.float32, [None, 2], name='x-input')
Y = tf.placeholder(tf.float32, [None, 1], name='y-input')

with tf.name_scope("layer1") as scope:  # 이렇게하면 layer1의 이름으로 분류될 수 있다.  즉 각 layer별로 정리하는 용도이다.
    W1 = tf.Variable(tf.random_normal([2, 2]), name='weight1')
    b1 = tf.Variable(tf.random_normal([2]), name='bias1')
    layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)

    w1_hist = tf.summary.histogram("weights1", W1)  # W1을 기록    histogram은 w1이 여러개의 dimension을 가지고있다는 의미이다.
    b1_hist = tf.summary.histogram("biases1", b1)   #b1을 기록
    layer1_hist = tf.summary.histogram("layer1", layer1)    #layer1을 기록


with tf.name_scope("layer2") as scope:
    W2 = tf.Variable(tf.random_normal([2, 1]), name='weight2')
    b2 = tf.Variable(tf.random_normal([1]), name='bias2')
    hypothesis = tf.sigmoid(tf.matmul(layer1, W2) + b2)

    w2_hist = tf.summary.histogram("weights2", W2)
    b2_hist = tf.summary.histogram("biases2", b2)
    hypothesis_hist = tf.summary.histogram("hypothesis", hypothesis)

# cost/loss function
with tf.name_scope("cost") as scope:
    cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) *
                           tf.log(1 - hypothesis))
    cost_summ = tf.summary.scalar("cost", cost) # scalar는 하나의 값을 가진다는 의미

with tf.name_scope("train") as scope:
    train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Accuracy computation
# True if hypothesis>0.5 else False
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))
accuracy_summ = tf.summary.scalar("accuracy", accuracy)

# Launch graph
with tf.Session() as sess:
    # tensorboard --logdir=./logs/xor_logs
    merged_summary = tf.summary.merge_all() # 다 합친다.
    writer = tf.summary.FileWriter("./logs/xor_logs_r0_01")   # 어디에 기록할것인지.
    writer.add_graph(sess.graph)  # 그래프를 넣어준다.

    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        summary, _ = sess.run([merged_summary, train], feed_dict={X: x_data, Y: y_data})    # summary를 실행한다.
        writer.add_summary(summary, global_step=step)   # 각 step마다 실제로 기록한다.

        if step % 100 == 0:
            print(step, sess.run(cost, feed_dict={
                  X: x_data, Y: y_data}), sess.run([W1, W2]))

    # Accuracy report
    h, c, a = sess.run([hypothesis, predicted, accuracy],
                       feed_dict={X: x_data, Y: y_data})
    print("\nHypothesis: ", h, "\nCorrect: ", c, "\nAccuracy: ", a)


# 이렇게 생성된 파일을 아나콘다 prompt에서 tensorboard -logdir="경로 및 파일명" 하고나서
# http://127.0.0.1:6006 이나 http://localhost:6006에 들어가면된다.
# 만약 learning rate를 다르게해줘서 비교하고싶다면 위처럼 파일을 2개만든 다음
# 경로 및 파일명에 그 파일들이 들어있는 디렉토리로 경로를 주면 비교가 가능하다.
# 본 코드의 경우 logs까지만 경로를 주면된다.