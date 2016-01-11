from tffactorization.tfnmf import TFNMF
import pandas as pd
from scipy.sparse import lil_matrix
import tensorflow as tf
import os


os.system("rm -rf logadagrad")

elist, com = pd.read_pickle("polblogs.pkl")
mat = lil_matrix((1222,1222))
for n1, n2 in elist:
    mat[n1,n2] = 1
    mat[n2, n1] = 1
V = mat.todense()
tfnmf = TFNMF(V,2, algo="grad", learning_rate=0.1)

sess = tf.InteractiveSession()
W, H = tfnmf.run(sess, logfile="logadagrad", max_iter=1000)

import tensorflow as tf
