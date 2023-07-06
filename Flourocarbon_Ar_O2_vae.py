import os

import numpy as np
import random as python_random


import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import backend as K

from tensorflow.keras import datasets, layers, models, regularizers, mixed_precision
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import (Dense, Conv1D, Conv2D, Input, Activation, BatchNormalization, 
                                     Reshape, Add, GlobalAveragePooling2D, GlobalAveragePooling1D, 
                                     Conv1DTranspose, Conv2DTranspose, LayerNormalization)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import TerminateOnNaN

import datetime
from pathlib import Path

import logging

from model_blocks import ConvNeXTBlock2D, ConvNeXTBlock1D, downsample_layer2D, downsample_layer1D, upsample_layer2D, upsample_layer1D, CosineDecayWarmup

def process_path(file_path):
    raw_img = tf.io.read_file(file_path + '/image.npy')
    img = tf.reshape(tf.io.decode_raw(raw_img, tf.float32)[32:], (96, 128, 3))
    
    raw_spectra = tf.io.read_file(file_path + '/spectra.npy')
    spectra = tf.reshape(tf.io.decode_raw(raw_spectra, tf.float32)[32:], (3072,1))

    return {'input_spectra':spectra, 'input_image':img}, {'spectra_decoded':spectra, 'image_decoded':img}
    # return {'input_spectra':spectra, 'input_image':img}

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        z_mean = tf.cast(z_mean, tf.float32)
        z_log_var = tf.cast(z_log_var, tf.float32)
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
# class VAE(keras.Model):
#     def __init__(self, encoder, image_decoder, spectra_decoder, beta, **kwargs):
#         super().__init__(**kwargs)
#         self.encoder = encoder
#         self.image_decoder = image_decoder
#         self.spectra_decoder = spectra_decoder
#         self.beta = beta
#         self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
#         self.image_reconstruction_loss_tracker = keras.metrics.Mean(
#             name="image_reconstruction_loss"
#         )
#         self.spectra_reconstruction_loss_tracker = keras.metrics.Mean(
#             name="spectra_reconstruction_loss"
#         )
#         self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

#     @property
#     def metrics(self):
#         return [
#             self.total_loss_tracker,
#             self.image_reconstruction_loss_tracker,
#             self.spectra_reconstruction_loss_tracker,
#             self.kl_loss_tracker,
#         ]

#     def train_step(self, data):
#         with tf.GradientTape() as tape:
#             # print(data)
#             # print(data[0]['input_spectra'])
#             # z_mean, z_log_var, z = self.encoder([data[0]['input_image'], data[0]['input_spectra']])
#             z_mean, z_log_var, z = self.encoder(data)
#             image_reconstruction = self.image_decoder(z)
#             spectra_reconstruction = self.spectra_decoder(z)
#             image_reconstruction_loss = tf.reduce_mean(
#                 tf.reduce_sum(
#                     keras.losses.mean_squared_error(data[0]['input_image'], image_reconstruction), axis=(1, 2)
#                 )
#             )
#             spectra_reconstruction_loss = tf.reduce_mean(
#                 tf.reduce_sum(
#                     keras.losses.mean_squared_error(data[0]['input_spectra'], spectra_reconstruction), axis=(1, 2)
#                 )
#             )
#             kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
#             kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
#             total_loss = image_reconstruction_loss + spectra_reconstruction_loss + (beta*kl_loss)
#         grads = tape.gradient(total_loss, self.trainable_weights)
#         self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
#         self.total_loss_tracker.update_state(total_loss)
#         self.image_reconstruction_loss_tracker.update_state(image_reconstruction_loss)
#         self.spectra_reconstruction_loss_tracker.update_state(spectra_reconstruction_loss)
#         self.kl_loss_tracker.update_state(kl_loss)
#         return {
#             "loss": self.total_loss_tracker.result(),
#             "image_reconstruction_loss": self.image_reconstruction_loss_tracker.result(),
#             "spectra_reconstruction_loss": self.spectra_reconstruction_loss_tracker.result(),
#             "kl_loss": self.kl_loss_tracker.result(),
#         }

# Create logging object

logger = logging.getLogger()

logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("'%(asctime)s.%(msecs)03d': %(message)s")

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

logging.info('Compute dtype: %s' % policy.compute_dtype)
logging.info('Variable dtype: %s' % policy.variable_dtype)

# when running on multiple gpus, this will stop tensorflow from automatically allocating itself all the available memory
# it will only use the memory needed, which is quite a polite thing to do on shared servers

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        logger.critical(e)

# uncomment 67-79 for more repeatable experiments, but fixing the random seed

# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.
# np.random.seed(123)

# # The below is necessary for starting core Python generated random numbers
# # in a well-defined state.
# python_random.seed(123)

# # The below set_seed() will make random number generation
# # in the TensorFlow backend have a well-defined initial state.
# # For further details, see:
# # https://www.tensorflow.org/api_docs/python/tf/random/set_seed
# tf.random.set_seed(1234)

# If you have stored the underlying dataset a different drive please ammend the path

path = Path('/storage/scratch_1/AEDatav2/flourocarbon_data_set/')

list_ds_train = tf.data.Dataset.list_files(str(path/'data_in_files_2nd_split/train_paired*'), shuffle=False)
list_ds_train = list_ds_train.shuffle(tf.data.experimental.cardinality(list_ds_train).numpy(), reshuffle_each_iteration=False)

list_ds_test = tf.data.Dataset.list_files(str(path/'data_in_files_2nd_split/test_paired*'), shuffle=False)
list_ds_test = list_ds_test.shuffle(tf.data.experimental.cardinality(list_ds_test).numpy(), reshuffle_each_iteration=False)

list_ds_val = tf.data.Dataset.list_files(str(path/'data_in_files_2nd_split/val_paired*'), shuffle=False)
list_ds_val = list_ds_val.shuffle(tf.data.experimental.cardinality(list_ds_val).numpy(), reshuffle_each_iteration=False)

epochs = 100
batch_size = 128 *16
spectra_length = 3072
image_array_shape = (1, 96, 128, 3)
channels = 1

AUTOTUNE = tf.data.AUTOTUNE

list_ds_train = list_ds_train.map(process_path, num_parallel_calls=AUTOTUNE)
list_ds_test = list_ds_test.map(process_path, num_parallel_calls=AUTOTUNE)
list_ds_val = list_ds_val.map(process_path, num_parallel_calls=AUTOTUNE)

def configure_for_performance(ds, batch_size):
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds

list_ds_train = configure_for_performance(list_ds_train, batch_size)
list_ds_test = configure_for_performance(list_ds_test, batch_size)
list_ds_val = configure_for_performance(list_ds_val, batch_size)

kern_reg = 1e-6
bias_reg = 1e-6

kern_reg_latent = 1e-6
bias_reg_latent = 1e-6

warmup_epoch = int(8)

total_steps = int(epochs * batch_size / len(gpus))

warmup_steps = int(warmup_epoch * batch_size / len(gpus))

dense_units_encoder = 32
dense_units_decoder = 32

# latent_units = 32

activation_fn = tf.nn.swish

initial_learning_rate = 1e-3
weight_decay = 0.005

drop_path_rate = 0

layer_scale_init_value = 0

blocks_spec = [2, 2, 6, 2]
blocks_spec_dec = [2, 6, 2, 2]
filters_spec = [64, 128, 256, 512]
filters_spec_dec = [512, 256, 128, 64]

depth_drop_rates = [
        float(x) for x in np.linspace(0.0, drop_path_rate, sum(blocks_spec))
    ]

kernel_initializer = tf.keras.initializers.GlorotUniform()
bias_initializer = tf.keras.initializers.Zeros()

# please ammend for the number of GPUs available

mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1", "/gpu:2", "/gpu:3"])

latent_units = 64

reg_latent = 1e-4

latent_norm = 'ln'

for beta in [1e-6]: # ['ln', 'bn']

    # on your own dataset, you may wish to test different initial learning rates and number of latent units, these are important hyperparameters

    for initial_learning_rate in [2.5e-4]:

        for latent_units in [latent_units]:

            repeat = 1
        # for repeat in range (1):
    
            with mirrored_strategy.scope():  

                # Encoder ----------------------------------------

                input_spectra = Input(shape=(spectra_length, channels), name = 'input_spectra')

                input_image = keras.Input(shape=(96, 128, 3), name = 'input_image')

                # stem

                spectra_encoder = Conv1D(
                                        filters_spec[0],
                                        kernel_size = 4,
                                        strides = 4,
                                        name = "spectra_encoder_stem_conv",
                                        )(input_spectra)

                spectra_encoder = LayerNormalization(epsilon=1e-6, name="spectra_encoder_stem_layernorm")(spectra_encoder)

                current_drop_rate = 0

                for i in range(len(blocks_spec)):

                    spectra_encoder = downsample_layer1D(filters_spec[i], i)(spectra_encoder)

                    for j in range(blocks_spec[i]):

                        spectra_encoder = ConvNeXTBlock1D(
                                                        filters_spec=filters_spec[i],
                                                        drop_path_rate=depth_drop_rates[current_drop_rate + j],
                                                        layer_scale_init_value=layer_scale_init_value,
                                                        name="spectra_encoder" + f"_stage_{i}_block_{j}",
                                                        )(spectra_encoder)

                    current_drop_rate += blocks_spec[i]

                spectra_encoder = GlobalAveragePooling1D(name = 'specra_GAP')(spectra_encoder)

                spectra_encoder = Dense(1024, 
                                        activation = activation_fn, 
                                        kernel_initializer=kernel_initializer, 
                                        bias_initializer=bias_initializer, 
                                        kernel_regularizer = l2(kern_reg), 
                                        bias_regularizer = l2(bias_reg), 
                                        name='spectra_dense_encoder'
                                        )(spectra_encoder)
                
                spectra_encoded = Dense(latent_units,
                                        activation = activation_fn, 
                                        kernel_initializer=kernel_initializer, 
                                        bias_initializer=bias_initializer, 
                                        kernel_regularizer = l2(kern_reg), 
                                        bias_regularizer = l2(bias_reg), 
                                        name='latent_spectra'
                                        )(spectra_encoder)
                
                image_encoder = Conv2D(
                                    filters_spec[0],
                                    kernel_size = 4,
                                    strides = 2,
                                    padding = 'same',
                                    name = "image_encoder_stem_conv",
                                    )(input_image)

                image_encoder = LayerNormalization(epsilon=1e-6, name="image_encoder_stem_layernorm")(image_encoder)

                current_drop_rate = 0

                for i in range(len(blocks_spec)):
                    
                    print(i)

                    image_encoder = downsample_layer2D(filters_spec[i], i)(image_encoder)

                    for j in range(blocks_spec[i]):

                        image_encoder = ConvNeXTBlock2D(
                                                        filters_spec=filters_spec[i],
                                                        drop_path_rate=depth_drop_rates[current_drop_rate + j],
                                                        layer_scale_init_value=layer_scale_init_value,
                                                        name="image_encoder" + f"_stage_{i}_block_{j}",
                                                        )(image_encoder)

                    current_drop_rate += blocks_spec[i]

                image_encoded = GlobalAveragePooling2D(name = 'image_GAP')(image_encoder)

                image_encoded = Dense(1024, 
                                      activation = activation_fn, 
                                      kernel_initializer=kernel_initializer, 
                                      bias_initializer=bias_initializer, 
                                      kernel_regularizer = l2(kern_reg), 
                                      bias_regularizer = l2(bias_reg), 
                                      name='image_dense_encoder'
                                      )(image_encoded)
                
                image_encoded = Dense(latent_units,
                                      activation = activation_fn, 
                                      kernel_initializer=kernel_initializer, 
                                      bias_initializer=bias_initializer, 
                                      kernel_regularizer = l2(kern_reg), 
                                      bias_regularizer = l2(bias_reg), 
                                      name='latent_image'
                                      )(image_encoded)

                combined_encoder = Add()([spectra_encoded, image_encoded])

                encoded_mean = Dense(latent_units, 
                                activation = None, 
                                dtype='float32', 
                                kernel_initializer=kernel_initializer, 
                                bias_initializer=bias_initializer, 
                                kernel_regularizer = l2(kern_reg), 
                                bias_regularizer = l2(bias_reg), 
                                name='latent_mean'
                                )(combined_encoder)
                
                encoded_log_var = Dense(latent_units, 
                                activation = None, 
                                dtype='float32', 
                                kernel_initializer=kernel_initializer, 
                                bias_initializer=bias_initializer, 
                                kernel_regularizer = l2(kern_reg), 
                                bias_regularizer = l2(bias_reg), 
                                name='latent_var'
                                )(combined_encoder)
                
                encoded = Sampling()([encoded_mean, encoded_log_var])

                # if latent_norm == 'bn':

                #     encoded = BatchNormalization(axis = -1, name = 'BN_encoder')(encoded)

                # elif latent_norm == 'ln':

                #     encoded = LayerNormalization(epsilon=1e-6, name="LN_encoder")(encoded)

                # else:

                #     encoded = Activation('linear')(encoded) 

                encoder = keras.Model([input_image, input_spectra], encoded)

                # Decoder ----------------------------------------

                spectra_decoder = Dense(latent_units, 
                                        activation = activation_fn, 
                                        kernel_initializer=kernel_initializer, 
                                        bias_initializer=bias_initializer,
                                        kernel_regularizer = l2(kern_reg), 
                                        bias_regularizer = l2(bias_reg), 
                                        name='spectra_dense_decoder'
                                        )(encoded)

                spectra_decoder = Dense(48, 
                                        activation = activation_fn, 
                                        kernel_initializer=kernel_initializer, 
                                        bias_initializer=bias_initializer, 
                                        kernel_regularizer = l2(kern_reg), 
                                        bias_regularizer = l2(bias_reg), 
                                        name = 'spectra_reform_dense'
                                        )(spectra_decoder)

                spectra_decoder = Reshape((48,1), name = 'spectra_reform_reshape')(spectra_decoder)

                spectra_decoder = Conv1D(128, 1, strides = 1, 
                                         padding='valid', 
                                         activation=activation_fn, 
                                         kernel_initializer=kernel_initializer, 
                                         bias_initializer=bias_initializer, 
                                         kernel_regularizer = l2(kern_reg), 
                                         bias_regularizer = l2(bias_reg), 
                                         name = 'spec_reform_conv'
                                         )(spectra_decoder)

                current_drop_rate = 0

                for i in range(len(blocks_spec_dec)):

                    spectra_decoder = upsample_layer1D(filters_spec_dec[i], i)(spectra_decoder)

                    for j in range(blocks_spec_dec[i]):

                        spectra_decoder = ConvNeXTBlock1D(
                                                        filters_spec=filters_spec_dec[i],
                                                        drop_path_rate=depth_drop_rates[::-1][current_drop_rate + j],
                                                        layer_scale_init_value=layer_scale_init_value,
                                                        name="spectra_decoder" + f"_stage_{i}_block_{j}",
                                                        )(spectra_decoder)

                    current_drop_rate += blocks_spec[i]

                spectra_decoder = Conv1DTranspose(
                                        filters_spec[0],
                                        kernel_size = 4,
                                        strides = 4,
                                        name = "spectra_decoder_destem_conv",
                                        )(spectra_decoder)

                spectra_decoder = LayerNormalization(epsilon=1e-6, name="spectra_decoder_stem_layernorm")(spectra_decoder)

                spectra_decoded = Conv1D(1, 3, 
                                         activation = 'sigmoid', 
                                         dtype='float32', 
                                         padding = 'same', 
                                         kernel_initializer=kernel_initializer, 
                                         bias_initializer=bias_initializer, 
                                         kernel_regularizer=l2(kern_reg), 
                                         bias_regularizer=l2(bias_reg), 
                                         name='spectra_decoded'
                                         )(spectra_decoder)
                
                image_decoder = Dense(latent_units*2, 
                                      activation = activation_fn, 
                                      kernel_initializer=kernel_initializer, 
                                      bias_initializer=bias_initializer, 
                                      kernel_regularizer = l2(kern_reg), 
                                      bias_regularizer = l2(bias_reg), 
                                      name='image_dense_decoder'
                                      )(encoded)
                
                image_decoder = Dense(image_encoder.shape[3], 
                                      activation = activation_fn, 
                                      kernel_initializer=kernel_initializer, 
                                      bias_initializer=bias_initializer, 
                                      kernel_regularizer = l2(kern_reg), 
                                      bias_regularizer = l2(bias_reg), 
                                      name = 'image_reform_dense'
                                      )(image_decoder)
                
                image_decoder = Reshape((1,1,image_encoder.shape[3]), input_shape = (image_encoder.shape[3],), name = 'image_diff_reform_reshape')(image_decoder)
                
                image_decoder = Conv2DTranspose(image_encoder.shape[3], 
                                                [image_encoder.shape[1], image_encoder.shape[2]], 
                                                strides = 2, 
                                                padding='valid', 
                                                activation=activation_fn, 
                                                kernel_initializer=kernel_initializer, 
                                                bias_initializer=bias_initializer, 
                                                kernel_regularizer = l2(kern_reg), 
                                                bias_regularizer = l2(bias_reg), 
                                                name = 'image_reform_conv'
                                                )(image_decoder)

                current_drop_rate = 0

                for i in range(len(blocks_spec_dec)):

                    image_decoder = upsample_layer2D(filters_spec_dec[i], i)(image_decoder)

                    for j in range(blocks_spec_dec[i]):

                        image_decoder = ConvNeXTBlock2D(
                                                        filters_spec=filters_spec_dec[i],
                                                        drop_path_rate=depth_drop_rates[::-1][current_drop_rate + j],
                                                        layer_scale_init_value=layer_scale_init_value,
                                                        name="image_decoder" + f"_stage_{i}_block_{j}",
                                                        )(image_decoder)

                    current_drop_rate += blocks_spec[i]

                image_decoder = Conv2DTranspose(
                                        filters_spec[0],
                                        kernel_size = 4,
                                        strides = 2,
                                        padding = 'same', 
                                        name = "image_decoder_destem_conv",
                                        )(image_decoder)

                image_decoder = LayerNormalization(epsilon=1e-6, name="image_decoder_stem_layernorm")(image_decoder)

                image_decoded = Conv2D(3, 3, 
                                       activation = 'sigmoid', 
                                       dtype='float32', 
                                       padding = 'same', 
                                       kernel_initializer=kernel_initializer, 
                                       bias_initializer=bias_initializer, 
                                       kernel_regularizer=l2(kern_reg), 
                                       bias_regularizer=l2(bias_reg), 
                                       name='image_decoded'
                                       )(image_decoder)

                autoencoder = keras.Model(inputs = [input_spectra, input_image], 
                                          outputs = [spectra_decoded, image_decoded])
                
                spectra_decoder_model = keras.Model(autoencoder.get_layer('spectra_dense_decoder').input, autoencoder.get_layer('spectra_decoded').output)
                image_decoder_model = keras.Model(autoencoder.get_layer('image_dense_decoder').input, autoencoder.get_layer('image_decoded').output)

                lr_schedule = CosineDecayWarmup(initial_learning_rate, total_steps, warmup_steps, alpha=1e-5)
                opt = Adam(learning_rate=lr_schedule)

                def KLLoss(z_log_var, z_mean):
                    kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
                    kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
                    return kl_loss
                
                def reconstruction_loss_image(input_image, image_decoded):
                    return tf.reduce_mean(tf.keras.metrics.mean_squared_error(input_image, image_decoded))
                    # return tf.keras.metrics.mean_squared_error(input_image, image_decoded)
                
                def reconstruction_loss_spectra(input_spectra, spectra_decoded):
                    return tf.reduce_mean(tf.keras.metrics.mean_squared_error(input_spectra, spectra_decoded))
                
                def reconstruction_loss(input, output):
                    print(tf.shape(input))
                    print(tf.shape(output))
                    return tf.reduce_mean(tf.keras.metrics.mean_squared_error(input, output))
                

                # def reconstruction_loss(input_spectra, input_image, spectra_decoded, image_decoded):
                #     return tf.reduce_mean(tf.reduce_sum(tf.keras.metrics.mean_squared_error(input_image, image_decoded), axis = (1, 2))) + tf.reduce_mean(tf.reduce_sum(tf.keras.metrics.mean_squared_error(input_spectra, spectra_decoded), axis = (1)))
                    # return tf.keras.metrics.mean_squared_error(input_spectra, spectra_decoded)
                
                # mse = keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)
                
                # reconstruction_loss_image = mse(input_image, image_decoded)

                # reconstruction_loss_spectra = mse(input_spectra, spectra_decoded)

                # print(tf.shape(reconstruction_loss_image(input_image, image_decoded)), reconstruction_loss_image(input_image, image_decoded))

                # print(tf.shape(reconstruction_loss_spectra(input_spectra, spectra_decoded)), reconstruction_loss_spectra(input_spectra, spectra_decoded))

                # print(tf.shape(KLLoss(encoded_log_var, encoded_mean)), KLLoss(encoded_log_var, encoded_mean))

                vae_loss = reconstruction_loss_image(input_image, image_decoded) + reconstruction_loss_spectra(input_spectra, spectra_decoded) + (beta * KLLoss(encoded_log_var, encoded_mean))

                autoencoder.add_loss(vae_loss)

                # break

                autoencoder.compile(optimizer = opt,
                        # loss = vae_loss,
                        steps_per_execution = 1,
                        jit_compile=False,
                        metrics = ['mean_absolute_error', 'mean_squared_error', 'binary_crossentropy', KLLoss, reconstruction_loss])

                # autoencoder = VAE(encoder, image_decoder_model, spectra_decoder_model, beta)

                # autoencoder.compile(optimizer = opt)

                # plot_model(autoencoder, to_file = 'vae.png', show_shapes=True, expand_nested=True)

                # uncomment to print a model summary to terminal, warning - it's very long

                # autoencoder.summary()

            now = datetime.datetime.now().strftime("%Y%m%d")
            logdir = os.path.join("logs/Image+Spectra_f_ar_o2_ConvNext-vT_vae_" + now, datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+ 'repeat_' + str(repeat) +'initial_lr='+str(initial_learning_rate)+'latent_reg='+str(reg_latent)+'_latent='+str(latent_units)+'_batch_size='+str(batch_size)+'latent_norm='+latent_norm)
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logdir, histogram_freq=1, profile_batch=0, update_freq = 'epoch')

            checkpoint_path = ('training_checkpoints/Image+Spectra_f_ar_o2_ConvNext-vT_vae_repeat_' + str(repeat) +'initial_lr='+str(initial_learning_rate)+'latent_reg='+str(reg_latent) + '_bn_latent='+str(latent_units)+'_batch_size='+str(batch_size)+'latent_norm='+latent_norm + now + '/epoch_val {epoch:02d}-{val_loss:.2f} .ckpt')
            
            checkpoint_dir = os.path.dirname(checkpoint_path)

            cp_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path,
                monitor = 'val_loss',
                verbose=1,
                save_best_only=False, 
                save_weights_only=True,
                save_frequency=10)

            term = TerminateOnNaN()

            callbacks_list = [tensorboard_callback, cp_callback, term]

            # if finetuning a model, pre-load the the weights and apply, uncomment and ammend the path to your original trained model

            # autoencoder_pre_trained = tf.keras.models.load_model('autoencoder_l128', compile=False)

            # autoencoder.set_weights(autoencoder_pre_trained.get_weights())

            # assert len(autoencoder_pre_trained.weights) == len(autoencoder.weights)
            # for a, b in zip(autoencoder_pre_trained.weights, autoencoder.weights):
            #     np.testing.assert_allclose(a.numpy(), b.numpy())

            history = autoencoder.fit(list_ds_train,
                    validation_data=list_ds_val,
                    epochs = epochs,
                    callbacks=callbacks_list)

            autoencoder.save('saved_model/f_ar_o2_image_spectra_vae_ConvNext-vT_repeat_' + str(repeat) + '_mse_bn_latent='+str(latent_units)+'_batch_size='+str(batch_size)+'_initial_lr='+str(initial_learning_rate)+'beta='+str(beta))
            encoder.save('saved_model/f_ar_o2_image_spectra_vae_encoder_ConvNext-vT_repeat_' + str(repeat) + '_mse_bn_latent='+str(latent_units)+'_batch_size='+str(batch_size)+'_initial_lr='+str(initial_learning_rate)+'beta='+str(beta))
            spectra_decoder_model.save('saved_model/f_ar_o2_spectra_vae_decoder_ConvNext-vT_repeat_' + str(repeat) + '_mse_bn_latent='+str(latent_units)+'_batch_size='+str(batch_size)+'_initial_lr='+str(initial_learning_rate)+'beta='+str(beta))
            image_decoder_model.save('saved_model/f_ar_o2_image_vae_decoder_ConvNext-vT_repeat_' + str(repeat) + '_mse_bn_latent='+str(latent_units)+'_batch_size='+str(batch_size)+'_initial_lr='+str(initial_learning_rate)+'beta='+str(beta))
            
            # when running in a loop, you can encounter memory leaks from improperly cleared tf graphs in gpu memory. Sometimes this helps to stop that, no guarantees.

            try:

                tf.keras.backend.clear_session()

            except:
                logging.CRITICAL('clear backend failed')
