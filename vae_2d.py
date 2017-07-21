import numpy as np
import numpy.random as rand
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime
from keras import backend as K
from vae import VAE
from scipy.stats import multivariate_normal
from scipy.stats import norm

# TODO Make comprehensive plots

z_start = -2
z_stop = 2

def plot_grad_vs_density(num_samples=100):
    n = num_samples
    #x0 = data_mean[0]
    #y0 = data_mean[1]

    #x1 = data_mean2[0]
    #y1 = data_mean2[1]

    #x_range = np.linspace(x0, x1, num=num_samples)
    #l = (y1 - y0) / (x1 - x0)
    #y_range = y0 + l*(x_range-x0)
    #x_test = np.asarray([i for i in zip(x_range, y_range)])

    #z = sess.run(vae.z, feed_dict={vae.x: x_test})

    z = np.linspace(start=z_start, stop=z_stop, num=n)
    z = z.reshape(n, 1)
    x_gen = sess.run(vae.x_hat, feed_dict={vae.z: z})

    x_real = next_batch(n)

    grad_x_z = sess.run(vae.grad_x_z, feed_dict={vae.z: z})[0]

    densities = multivariate_normal.pdf(x_gen, mean=data_mean, cov=np.eye(data_dim)*data_std) + multivariate_normal.pdf(x_gen, mean=data_mean2, cov=np.eye(data_dim)*data_std2)

    sort_ind = np.argsort(densities)

    grad_norm = np.linalg.norm(grad_x_z, ord=2, axis=1)

    #plt.figure(2)
    #plt.plot(densities[sort_ind], grad_norm[sort_ind])
    #plt.xlabel("density")
    #plt.ylabel("gradient")
    #plt.show()

    #plt.figure(4)
    #plt.plot(densities, '.')
    #plt.xlabel("z")
    #plt.ylabel("x_gen pdf")
    ##plt.savefig("x_gen_pdf.svg")
    ##plt.show()

    plt.figure(3)
    plt.plot(grad_norm)
    plt.xlabel("z")
    plt.ylabel("gradient x_hat w.r.t z")
    #plt.savefig("gradient_xhat_z.svg")
    plt.show()
    plt.close()

def plot_points(n=1000):
    z = np.linspace(start=z_start, stop=z_stop, num=n)
    z = z.reshape(n, 1)
    #z = np.random.normal(size=(n, latent_size), loc=0, scale=1)
    #x_gen = sess.run(vae.generate_sample, feed_dict={vae.z: z})

    #x0 = data_mean[0]
    #y0 = data_mean[1]

    #x1 = data_mean2[0]
    #y1 = data_mean2[1]

    #x_range = np.linspace(x0, x1, num=n)
    #l = (y1 - y0) / (x1 - x0)
    #y_range = y0 + l*(x_range-x0)
    #x_test = np.asarray([i for i in zip(x_range, y_range)])

    #z = sess.run(vae.z, feed_dict={vae.x: x_test})
    x_gen = sess.run(vae.generate_sample, feed_dict={vae.z: z})
    x_real = next_batch(n)

    plt.figure(1)
    plt.scatter(x=x_real[:, 0], y=x_real[:, 1], label='x_real')
    plt.scatter(x=x_gen[:, 0], y=x_gen[:, 1], label='x_gen')
    #plt.scatter(x=x_test[:, 0], y=x_test[:, 1], label='x_test')
    plt.title("Varying z from %d to %d" % (z_start, z_stop))
    plt.grid()
    plt.legend(loc='upper right')
    #plt.savefig("x_gen_vs_z.svg")
    #plt.show()
    #plt.close()


def next_batch2(batch_size):
    z_true = np.random.uniform(0, 1, batch_size)
    r = np.power(z_true, 0.5)
    phi = 0.25 * np.pi * z_true
    mu1 = r * np.cos(phi)
    mu2 = r * np.sin(phi)
    std1 = 0.10 * np.power(z_true, 2)
    std2 = 0.10 * np.power(z_true, 2)

    # Sampling form a Gaussian
    x1 = np.random.normal(mu1, std1, batch_size)
    x2 = np.random.normal(mu2, std2, batch_size)

    # Bringing data in the right form
    X = np.transpose(np.random.normal(
        [mu1, mu2], [std1, std2], (2, batch_size)))

    return X


def next_batch(batch_size=100):
    data_mean = [0, 10]
    data_std = [1, 1]
    X1 = np.random.normal(data_mean, data_std, (int(batch_size / 2), 2))
    X2 = np.random.normal(data_mean2, data_std2, (int(batch_size / 2), 2))
    return np.concatenate((X1,X2))

    #X1 = np.random.normal(data_mean, data_std, (batch_size, 2))
    #return X1


data_mean = [0, 10]
data_std = [1, 1]
data_mean2 = [-5, 5]
data_std2 = [1, 1]
data_dim = 2
latent_size = 1
batch_size = 100
num_iter = 10**4

vae = VAE(latent_size=latent_size, input_size=data_dim, hidden_units=5)

sess = tf.InteractiveSession()
K.set_session(sess)
train_writer = tf.summary.FileWriter(
    './train/' + str(datetime.now()), sess.graph)
tf.global_variables_initializer().run()

for i in range(num_iter):
    x_real = next_batch(batch_size)
    summary, _ = sess.run([vae.step_summary, vae.train_step], feed_dict={
                          vae.x: x_real})
    if i % 100 == 0:
        print(i)
        train_writer.add_summary(summary, i)

plot_points()
plot_grad_vs_density(100)
