import numpy as np
import tensorflow as tf
import numpy.random as rand
from datetime import datetime
import matplotlib.pyplot as plt
from keras import backend as K
from vae import VAE

data_mean = (0, 100)
data_std = (1, 5)
data_dim = 2
latent_size = 2
batch_size = 100
num_iter = 20000

vae = VAE(latent_size=latent_size, input_size=data_dim, hidden_units=10)

sess = tf.InteractiveSession()
K.set_session(sess)
train_writer = tf.summary.FileWriter('./train/'+str(datetime.now()), sess.graph)
tf.global_variables_initializer().run()

for i in range(num_iter):
    print(i)
    x_real = rand.normal(size=(batch_size, data_dim), loc=data_mean, scale=data_std)
    eps = rand.normal(size=(batch_size, latent_size), loc=0, scale=1)
    summary, _ = sess.run([vae.step_summary, vae.train_step], feed_dict={vae.x:x_real, vae.eps:eps})
    if i % 100 == 0:
        train_writer.add_summary(summary, i)

num_samples = 1000
z = np.random.normal(size=(num_samples, latent_size), loc=0, scale=1)
samples = sess.run(vae.generate_sample, feed_dict={vae.z:z})
x_real = rand.normal(size=(num_samples, data_dim), loc=data_mean, scale=data_std)
eps = np.random.normal(size=(num_samples, latent_size), loc=0, scale=1)
x_hat = sess.run(vae.x_hat, feed_dict={vae.x:x_real, vae.eps:eps})

np.save("samples", samples)
np.save("x_real", x_real)
np.save("x_hat", x_hat)
