import tensorflow as tf

#placeholder라는 node만든다.
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a+b
#그래프 완성

sess = tf.Session()

#feed_dict 으로 값을 넘겨준다.
print(sess.run(adder_node, feed_dict={a:3, b:4.5}))
print(sess.run(adder_node, feed_dict={a:[1,3], b:[2,4]}))