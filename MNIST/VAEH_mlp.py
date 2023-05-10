"""
Title: Variational AutoEncoder
Author: [fchollet](https://twitter.com/fchollet)
Date created: 2020/05/03
Last modified: 2020/05/03
Description: Convolutional Variational AutoEncoder (VAE) trained on MNIST digits.
"""

"""
## Setup
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from explainer import explain_integrated_gradiant, explain_cem
import vectorization_tools
from sample import Sample
from tensorflow.keras import backend as K

"""
## Create a belin of sampling layer
"""

L_DIM = 200
I_DIM = 400

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


img_dim = 28
img_chn = 1

"""
## Build the encoder
"""

original_dim = img_dim*img_dim*img_chn
latent_dim = L_DIM
intermediate_dims = np.array([I_DIM])

encoder_inputs = tf.keras.Input(shape=(original_dim,))
x = layers.Dense(intermediate_dims[0], activation="relu")(encoder_inputs)
x = layers.Dense(intermediate_dims[0], activation="relu")(x)
z_mean = layers.Dense(latent_dim, name='z_mean')(x)
z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)
z = Sampling()([z_mean, z_log_var])
encoder = tf.keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
encoder.summary()

"""
## Build the decoder
"""

intermediate_dims = np.flipud(intermediate_dims)
latent_inputs = tf.keras.Input(shape=(latent_dim,))
x = layers.Dense(intermediate_dims[0], activation='relu')(latent_inputs)
x = layers.Dense(intermediate_dims[0], activation='relu')(x)
pos_mean = layers.Dense(original_dim, name='pos_mean', activation='sigmoid')(x)

decoder = tf.keras.Model(latent_inputs, pos_mean, name="decoder")
decoder.summary()


"""
## Define the VAE as a `Model` with a custom `train_step`
"""


class VAE(tf.keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            outputs = self.decoder(z)

            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    # TODO: mse is worse
                    tf.keras.losses.binary_crossentropy(data, outputs), axis=(-1)
                    # tf.keras.losses.mean_squared_error(data, outputs), axis=(-1)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=(-1)))
            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


"""
## Train the VAE
"""

mnist_digits = np.load('dataset_mnist_heatmap.npy')
# min: 0.0 max: 0.64
mnist_digits = np.where(mnist_digits > 0.01, 1, mnist_digits)
mnist_digits = np.expand_dims(mnist_digits, -1)


vae_name = "models/mnist_mlp"
mnist_digits = tf.reshape(tensor=mnist_digits, shape=(-1, original_dim,))

vae = VAE(encoder, decoder)

vae.compile(optimizer="adam")
vae.fit(mnist_digits, epochs=100, batch_size=128)

vae.encoder.save(vae_name+"/encoder_heatmap_200")
vae.decoder.save(vae_name+"/decoder_heatmap_200")