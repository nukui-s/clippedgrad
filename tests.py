import numpy as np
from sklearn.decomposition import NMF
import tensorflow as tf
from clippedgrad import ClippedAdagradOptimizer
from clippedgrad import ClippedGDOptimizer
import time
import os
from sklearn.datasets import load_digits
from tffactorization.tfnmf import TFNMF
from sklearn.datasets import load_sample_images

os.system("rm -rf logtest")
sess = tf.InteractiveSession()
writer = tf.train.SummaryWriter("logtest",sess.graph_def)
N = 8
K = 2
v = np.random.random(size=[N,N]).astype(np.float32)
v = load_sample_images().images[0][0:400,0:200,0]
N = v.shape[0]
M = v.shape[1]
w = np.random.rand(N,K).astype(np.float32)
h = np.random.rand(K,M).astype(np.float32)

tfnmf = TFNMF(v,K)
start = time.time()
W3, H3 = tfnmf.run(sess)
end = time.time()
loss = np.power(v - np.matmul(W3,H3),2).sum() / (M*N)
print(end-start)
print("loss:",loss)

V = tf.placeholder("float", shape=[N,M])
W = tf.Variable(w)
H = tf.Variable(h)
loss = tf.nn.l2_loss(V - tf.matmul(W,H))/(N*M)
tf.scalar_summary("loss", loss)
merged = tf.merge_all_summaries()
opt1 = ClippedAdagradOptimizer(1.0).minimize(loss)
opt2 = ClippedGDOptimizer(1.0).minimize(loss)
opt3 = tf.train.GradientDescentOptimizer(0.0001).minimize(loss)

tf.initialize_all_variables().run()
start = time.time()
ls_old = np.infty
for i in range(10000):
    W1, H1, sm, ls, _ = sess.run([W,H,merged,loss,opt1],feed_dict={V: v})
    writer.add_summary(sm, i)
    if ls_old - ls < 0.01:
        break
    ls_old = ls
end = time.time()
print(end-start)
loss = np.power(v - np.matmul(W1,H1),2).sum()/ (M*N)
print("loss:",loss)


nmf = NMF(K)
start = time.time()
W2 = nmf.fit_transform(v)
H2 = nmf.components_
end = time.time()
print(end-start)
loss = np.power(v - np.matmul(W2,H2),2).sum()/ (M*N)
print("loss:",loss)
