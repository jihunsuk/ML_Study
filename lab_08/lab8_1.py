# https://www.tensorflow.org/api_guides/python/array_ops
import tensorflow as tf
import numpy as np
import pprint
tf.set_random_seed(777)  # for reproducibility

pp = pprint.PrettyPrinter(indent=4)
sess = tf.InteractiveSession()

# Simple Array
t = np.array([0., 1., 2., 3., 4., 5., 6.])
pp.pprint(t)
print(t.ndim) # rank
print(t.shape) # shape
print(t[0], t[1], t[-1])
print(t[2:5], t[4:-1])
print(t[:2], t[3:])

# 2D Array
t = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [10., 11., 12.]])
pp.pprint(t)
print(t.ndim) # rank
print(t.shape) # shape

# Shape, Rank, Axis
t = tf.constant([1,2,3,4])
print(tf.shape(t).eval())

t = tf.constant([[1,2],
                 [3,4]])
print(tf.shape(t).eval())

t = tf.constant([[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],[[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]]])
print(tf.shape(t).eval())

# Matmul vs multiply의 차이
matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2.],[2.]])
print(tf.matmul(matrix1, matrix2).eval())

print((matrix1*matrix2).eval())

# Watch out braodcasting    braodcasting은 매트릭스의 구조가 달라도 연산이 되게하는것이다.
matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2.],[2.]])
print((matrix1+matrix2).eval())

matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2., 2.]])
print((matrix1+matrix2).eval())

# Random values for variable initializations
print(tf.random_normal([3]).eval())
print(tf.random_uniform([2]).eval())
print(tf.random_uniform([2, 3]).eval())

# Reduce Mean/Sum  평균을 다룰 때는 무조건 float로 해야한다.
print(tf.reduce_mean([1, 2], axis=0).eval())
x = [[1., 2.],
     [3., 4.]]
print(tf.reduce_mean(x).eval())
print(tf.reduce_mean(x, axis=0).eval()) # axis(축)이 0으로 평균을 구하라고하면 x의 1,3의 평균, 2,4의 평균
print(tf.reduce_mean(x, axis=1).eval()) # axis 1로 평균을 구하라고하면 1,2과 3,4의 평균이 구해진다.
print(tf.reduce_mean(x, axis=-1).eval()) # -1은 가장 안쪽의 axis이므로 1과 같다.
print(tf.reduce_sum(x).eval())
print(tf.reduce_sum(x, axis=0).eval())
print(tf.reduce_sum(x, axis=-1).eval())
print(tf.reduce_mean(tf.reduce_sum(x, axis=-1)).eval()) # 가장 안쪽의 합을 구하고 평균을 낸다.

# Argmax with axis  # argmax는 max값의 위치를 구하는것이다.
x = [[0, 1, 2],
     [2, 1, 0]]
print(tf.argmax(x, axis=0).eval())  # axis가 0이므로 0과2중 큰수, 1과 1중 큰수, 2와0중 큰수쪽의 index를 출력
print(tf.argmax(x, axis=1).eval())
print(tf.argmax(x, axis=-1).eval())

# Reshape **중요 - 가장 많이 사용한다
t = np.array([[[0, 1, 2],
               [3, 4, 5]],

              [[6, 7, 8],
               [9, 10, 11]]])
print(t.shape)

print(tf.reshape(t, shape=[-1, 3]).eval())  # 보통 가장 안쪽인 3은 보통 안건드린다. 그래서 원소의 개수는 같게유지된다.
print(tf.reshape(t, shape=[-1, 1, 3]).eval())

# Squeeze 밑에거를 예로하면 [0, 1, 2]로 묶어준다.
print(tf.squeeze([[0], [1], [2]]).eval())

# expand  밑을 예로들면 [[0], [1], [2]] 로 풀어버린다.
print(tf.expand_dims([0, 1, 2], 1).eval())

# One hot   deptth가 3이므로 3개를 분류할 수 있고 0번쨰, 1번째, 2번째, 0번째를 각각 one hot시킨다.
# One hot을 하면 rank가 1개늘어난다.
print(tf.one_hot([[0], [1], [2], [0]], depth=3).eval())
# 늘어나기 때문에 reshape으로 줄여줄 수 있다.
t = tf.one_hot([[0], [1], [2], [0]], depth=3)
print(tf.reshape(t, shape=[-1, 3]).eval())

# Casting
print(tf.cast([1.8, 2.2, 3.3, 4.9], tf.int32).eval())
print(tf.cast([True, False, 1 == 1, 0 == 1], tf.int32).eval())

# Stack
x = [1, 4]
y = [2, 5]
z = [3, 6]

# Pack along first dim.
print(tf.stack([x, y, z]).eval())  #x, y, z를 쌓아준다.
print(tf.stack([x, y, z], axis=-1).eval())

# 1또는 0채우기
x = [[0, 1, 2],
     [2, 1, 0]]
print(tf.ones_like(x).eval())
print(tf.zeros_like(x).eval())

# Zip
for x, y in zip([1, 2, 3], [4, 5, 6]):
    print(x, y)
for x, y, z in zip([1, 2, 3], [4, 5, 6], [7, 8, 9]):
    print(x, y, z)