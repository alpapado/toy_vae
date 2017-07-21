import tensorflow as tf
from keras.layers import Dense


class VAE:
    def __init__(self, latent_size=2, input_size=784, hidden_units=512):
        self.latent_size = latent_size
        self.input_size = input_size
        self.hidden_units = hidden_units
        self.build_model()

    def build_model(self):
        x = tf.placeholder(tf.float32, shape=[None, self.input_size])
        eps = tf.random_normal(shape=(tf.shape(x)[0], 1) )

        # Encoder
        with tf.name_scope("encoder"):
            h = Dense(units=self.hidden_units, activation='relu')(x)
            z_mu = Dense(units=self.latent_size, activation='linear')(h)
            z_log_sigma_squared = Dense(
                units=self.latent_size, activation='linear')(h)
            z_sigma = tf.exp(1 / 2 * z_log_sigma_squared)
            z = z_mu + tf.multiply(z_sigma, eps)

        # Decoder
        with tf.name_scope("decoder"):
            h2 = Dense(units=self.hidden_units, activation='relu')(z)
            x_mu = Dense(units=self.input_size, activation='linear')(h2)
            x_log_sigma_squared = Dense(
                units=self.input_size, activation='linear')(h2)
            x_hat = x_mu

        with tf.name_scope("loss"):
            kl_divergence = -0.5 * \
                tf.reduce_sum(1 + z_log_sigma_squared -
                              tf.square(z_mu) - tf.exp(z_log_sigma_squared), axis=1)
            reconstruction_error = tf.reduce_sum(
                0.5 * (x_log_sigma_squared + tf.square(x - x_mu) / tf.exp(x_log_sigma_squared) ), axis=1)

            #reconstruction_error = tf.losses.mean_squared_error(labels=x, predictions=x_hat)

            J = tf.reduce_mean(kl_divergence + reconstruction_error)
            tf.summary.scalar("Total_loss", tf.reduce_mean(J))
            tf.summary.scalar("KL_divergence", tf.reduce_mean(kl_divergence))
            tf.summary.scalar("Reconstruction_error",
                              tf.reduce_mean(reconstruction_error))
            train_step = tf.train.AdamOptimizer().minimize(J)
            #train_step = tf.train.RMSPropOptimizer().minimize(J)

        # Expose some variables
        self.x = x
        self.z = z
        self.z_mean = z_mu
        self.z_sigma = z_sigma
        self.eps = eps
        self.train_step = train_step
        self.step_summary = tf.summary.merge_all()
        self.x_hat = x_mu #+ tf.exp(1/2*x_log_sigma_squared)
        self.generate_sample = x_hat

        self.grad_x_z = tf.gradients(self.x_hat, self.z)
        self.grad_z_x = tf.gradients(self.z, self.x)
