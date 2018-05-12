import tensorflow as tf
hello = tf.constant("Hello, TensorFlow!")   #텐서플로우 노드 생성
sess = tf.Session()
print(sess.run(hello))  #노드실행

#b'Hello, TensorFlow!'라고 출력된다.  b라는것은 byte String이라는 의미