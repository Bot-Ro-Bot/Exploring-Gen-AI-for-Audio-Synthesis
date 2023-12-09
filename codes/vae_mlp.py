from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras import backend as K

class VAE_MLP:
    def __init__(self, original_dim, latent_dim):
        self.original_dim = original_dim
        self.latent_dim = latent_dim
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        self.vae = self.build_vae()

    def build_encoder(self):
        inputs = Input(shape=(self.original_dim,))
        x = Dense(1024, activation='relu')(inputs)
        x = Dense(512, activation='relu')(x)
        x = Dense(256, activation='relu')(x)
        z_mean = Dense(self.latent_dim, name='z_mean')(x)
        z_log_var = Dense(self.latent_dim, name='z_log_var')(x)

        def sampling(args):
            z_mean, z_log_var = args
            batch_size = K.shape(z_mean)[0]
            epsilon = K.random_normal(shape=(batch_size, self.latent_dim), mean=0., stddev=1.)
            return z_mean + K.exp(0.5 * z_log_var) * epsilon

        z = Lambda(sampling, output_shape=(self.latent_dim,), name='z')([z_mean, z_log_var])
        encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
        return encoder

    def build_decoder(self):
        decoder_inputs = Input(shape=(self.latent_dim,), name='decoder_input')
        x = Dense(256, activation='relu')(decoder_inputs)
        x = Dense(512, activation='relu')(x)
        x = Dense(1024, activation='relu')(x)
        outputs = Dense(self.original_dim, activation='sigmoid')(x)
        decoder = Model(decoder_inputs, outputs, name='decoder')
        return decoder

    def build_vae(self):
        inputs = Input(shape=(self.original_dim,))
        z_mean, z_log_var, z = self.encoder(inputs)
        outputs = self.decoder(z)
        
        def vae_loss(inputs, x_decoded_mean):
            recon_loss = self.original_dim * binary_crossentropy(inputs, x_decoded_mean)
            kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            return K.mean(recon_loss + kl_loss)

        vae = Model(inputs, outputs, name='vae')
        vae.compile(optimizer='adam', loss=vae_loss)
        return vae


original_dim = (257,69)
latent_dim = 3

mlp_vae = VAE_MLP(original_dim=original_dim, latent_dim=latent_dim)

