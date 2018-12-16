import variational_autoencoder_opt_util as vae_util
from variational_autoencoder_dfc_util import plot_image_rows
from keras import backend as K
from keras import layers
from keras.models import Model
from keras.models import load_model

def create_vae(latent_dim, return_kl_loss_op=False):
    '''Creates a VAE able to auto-encode MNIST images and optionally its associated KL divergence loss operation.'''

    encoder = vae_util.create_encoder(latent_dim)
    decoder = vae_util.create_decoder(latent_dim)
    sampler = vae_util.create_sampler()

    x = layers.Input(shape=(28, 28, 1), name='image')
    t_mean, t_log_var = encoder(x)
    t = sampler([t_mean, t_log_var])
    t_decoded = decoder(t)

    model = Model(x, t_decoded, name='vae')
    
    if return_kl_loss_op:
        kl_loss = -0.5 * K.sum(1 + t_log_var \
                                 - K.square(t_mean) \
                                 - K.exp(t_log_var), axis=-1)
        return model, kl_loss
    else:
        return model

def vae_loss(x, t_decoded):
    '''Total loss for the plain VAE'''
    return K.mean(reconstruction_loss(x, t_decoded) + vae_kl_loss)


def vae_dfc_loss(x, t_decoded):
    '''Total loss for the DFC VAE'''
    latent_dim = 5
    vae_dfc, vae_dfc_kl_loss = create_vae(latent_dim, return_kl_loss_op=True)
    return K.mean(perceptual_loss(x, t_decoded) + vae_dfc_kl_loss)


def reconstruction_loss(x, t_decoded):
    '''Reconstruction loss for the plain VAE'''
    return K.sum(K.binary_crossentropy(
        K.batch_flatten(x), 
        K.batch_flatten(t_decoded)), axis=-1)


def perceptual_loss(x, t_decoded):
    '''Perceptual loss for the DFC VAE'''
    # Load pre-trained preceptual model. A simple CNN for
    # classifying MNIST handwritten digits.
    pm = load_model('models/vae-opt/classifier.h5')

    # Names and weights of perceptual model layers 
    # selected for calculating the perceptual loss.
    selected_pm_layers = ['conv2d_6', 'conv2d_7']
    selected_pm_layer_weights = [1.0, 1.0]
	
    outputs = [pm.get_layer(l).output for l in selected_pm_layers]
    model = Model(pm.input, outputs)

    h1_list = model(x)
    h2_list = model(t_decoded)
    
    rc_loss = 0.0
    
    for h1, h2, weight in zip(h1_list, h2_list, selected_pm_layer_weights):
        h1 = K.batch_flatten(h1)
        h2 = K.batch_flatten(h2)
        rc_loss = rc_loss + weight * K.sum(K.square(h1 - h2), axis=-1)
    
    return rc_loss

def encode(model, images):
    '''Encodes images with the encoder of the given auto-encoder model'''
    return model.get_layer('encoder').predict(images)[0]


def decode(model, codes):
    '''Decodes latent vectors with the decoder of the given auto-encoder model'''
    return model.get_layer('decoder').predict(codes)


def encode_decode(model, images):
    '''Encodes and decodes an image with the given auto-encoder model'''
    return decode(model, encode(model, images))
