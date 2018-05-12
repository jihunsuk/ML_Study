# 파일이 많아 메모리가 부족할 수 있다. 그래서 파일을 나눠서 읽을 수 있다.  Queue Runners

import tensorflow as tf
tf.set_random_seed(777)  # for reproducibility

#Queue Runners를 만든다. 지금은 파일 1개니까 [파일이름 넣는다.] 여러개인경우 , 으로 여러개 가능
#파일 셔플안한다.
filename_queue = tf.train.string_input_producer(
    ['lab4_3.csv'], shuffle=False, name='filename_queue')

reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

# Default values, in case of empty columns. Also specifies the type of the
# decoded result.
record_defaults = [[0.], [0.], [0.], [0.]]  #읽어올 각각의 데이터타입은 float다
xy = tf.decode_csv(value, record_defaults=record_defaults)  #value를 csv로 decode하라

# collect batches of csv in
train_x_batch, train_y_batch = \
    tf.train.batch([xy[0:-1], xy[-1:]], batch_size=10)  #읽어온 xy를 10개씩 끊어서 가져온다.

# placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis
hypothesis = tf.matmul(X, W) + b

# Simplified cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

# Launch the graph in a session.
sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

# Start populating the filename queue.  여기는 무조건 그냥 하는부분
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

for step in range(2001):
    x_batch, y_batch = sess.run([train_x_batch, train_y_batch]) #펌프를해서 데이터를 계속 가져온다.
    cost_val, hy_val, _ = sess.run(
        [cost, hypothesis, train], feed_dict={X: x_batch, Y: y_batch})
    if step % 10 == 0:
        print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)

coord.request_stop()  #여기도 무조건 그냥 하는부분
coord.join(threads)