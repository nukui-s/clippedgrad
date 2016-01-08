import numpy as np
import tensorflow as tf
from clippedgrad import ClippedAdagradOptimizer

sess = tf.InteractiveSession()
v = -np.random.normal(size=[8,8]).astype(np.float32)
w = np.random.rand(8,2).astype(np.float32)
h = np.random.rand(2,8).astype(np.float32)

V = tf.placeholder("float", shape=[8,8])
W = tf.Variable(w)
H = tf.Variable(h)
loss = tf.nn.l2_loss(V - tf.matmul(W,H))
opt = ClippedAdagradOptimizer(0.1).minimize(loss)

tf.initialize_all_variables().run()
for i in range(100):
    W1, H1, _ = sess.run([W,H,opt],feed_dict={V: v})

print(w)
print(W1)
