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
        eps = tf.placeholder(tf.float32, shape=[None, self.latent_size])

        # Encoder
        with tf.name_scope("encoder"):
            h = Dense(units=self.hidden_units, activation='relu')(x)
            mu = Dense(units=self.latent_size, activation='linear')(h)
            log_sigma_squared = Dense(units=self.latent_size, activation='linear')(h)
            sigma = tf.exp(1/2 * log_sigma_squared)
            z = mu + tf.multiply(sigma, eps)

        # Decoder
        with tf.name_scope("decoder"):
            h2 = Dense(units=self.hidden_units, activation='relu')(z)
            x_hat = Dense(units=self.input_size, activation='linear')(h2)

        with tf.name_scope("loss"):
            kl_divergence = -0.5 * tf.reduce_sum( 1 + log_sigma_squared - tf.pow(mu, 2) - tf.exp(log_sigma_squared), axis=1 )
            #reconstruction_error = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=x_hat, labels=x), axis=1)
            reconstruction_error = tf.losses.mean_squared_error(labels=x, predictions=x_hat)
            #weight_decay = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name and 'decoder' in v.name]) * 0.001

            J = tf.reduce_mean(kl_divergence + reconstruction_error)
            tf.summary.scalar("Total loss", tf.reduce_mean(J))
            tf.summary.scalar("KL divergence", tf.reduce_mean(kl_divergence))
            tf.summary.scalar("Reconstruction error", tf.reduce_mean(reconstruction_error))
            train_step = tf.train.AdamOptimizer().minimize(J)

        #with tf.name_scope("x_hat_image"):
        #    x_hat_image = tf.reshape(tf.nn.sigmoid(x_hat), shape=[-1, 28, 28, 1])
        #    tf.summary.image("x_hat_image", x_hat_image)

        #with tf.name_scope("x_image"):
        #    x_image = tf.reshape(x, shape=[-1, 28, 28, 1])
        #    tf.summary.image("x_image", x_image)

        # Expose some variables
        self.x = x
        self.z = z
        self.z_mean = mu
        self.z_sigma = sigma
        self.eps = eps
        self.train_step = train_step
        self.step_summary = tf.summary.merge_all()
        self.x_hat = x_hat
        self.generate_sample = x_hat

