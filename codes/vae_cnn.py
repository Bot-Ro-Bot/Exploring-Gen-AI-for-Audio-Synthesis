from tensorflow.keras.layers import Input, Dropout, Conv2D, Flatten, Dense, Lambda, Reshape, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K


class VAE_CNN:
    def __init__(self, input_shape, latent_dim):
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        self.vae = self.build_vae()

    def build_encoder(self):
        inputs = Input(shape=self.input_shape)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = Flatten()(x)
        x = Dropout(0.2)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.2)(x)
        z_mean = Dense(self.latent_dim)(x)
        z_log_var = Dense(self.latent_dim)(x)

        def sampling(args):
            z_mean, z_log_var = args
            batch = K.shape(z_mean)[0]
            dim = K.int_shape(z_mean)[1]
            epsilon = K.random_normal(shape=(batch, dim), mean=0.0, stddev=1.0)
            return z_mean + K.exp(0.5 * z_log_var) * epsilon

        z = Lambda(sampling)([z_mean, z_log_var])
        encoder = Model(inputs, z)
        return encoder

    def build_decoder(self):
        decoder_input = Input(shape=(self.latent_dim,))
        y = Dense(256)(decoder_input)
        y = Dense(self.input_shape[0] * self.input_shape[1] * 64, activation='relu')(y)
        y = Reshape((self.input_shape[0], self.input_shape[1], 64))(y)
        y = Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(y)
        y = Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='same')(y)
        decoder_output = y
        decoder = Model(decoder_input, y)
        return decoder

    def build_vae(self):
        inputs = Input(shape=self.input_shape)
        z = self.encoder(inputs)
        outputs = self.decoder(z)
        vae = Model(inputs, outputs)

        def vae_loss(inputs, outputs):
            xent_loss = K.sum(K.binary_crossentropy(inputs, outputs), axis=(1, 2, 3))
            kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            return K.mean(xent_loss + kl_loss)

        vae.compile(optimizer='adam', loss=vae_loss)
        return vae


input_shape = (257, 69, 1)
latent_dim = 2

vae_cnn  = VAE_CNN(input_shape=input_shape, latent_dim=latent_dim)