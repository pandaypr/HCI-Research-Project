'''This is another code that can be used for Grad-CAM and generate heatmap but it does not use yield method for image generation, it uses simple way of generating images dataset
'''

import os
import numpy as np
import sys
from tqdm import tqdm
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape, Lambda, Activation,     BatchNormalization, LeakyReLU, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint 
import tensorflow.keras.utils 


# In[2]:


WEIGHTS_FOLDER = './weights/'
if not os.path.exists(WEIGHTS_FOLDER):
    os.makedirs(os.path.join(WEIGHTS_FOLDER, "AE"))
    os.makedirs(os.path.join(WEIGHTS_FOLDER, "VAE"))


# In[78]:


IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 1
INPUT_DIM = (128,128,1)
BATCH_SIZE = 16
Z_DIM = 10
TRAIN_PATH='SmallSet/images/train/image'
ANNOT_PATH='SmallSet/images/train/label'
#TEST_PATH = '../input/stage1_test/'

train_ids = list(int(s.split(".")[0]) for s in list(next(os.walk(TRAIN_PATH))[2]))
len(train_ids)


# In[79]:


X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1))
print('Getting and resizing train images and masks ... ')
sys.stdout.flush()
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
  img = imread(TRAIN_PATH+'/'+str(id_)+'.jpg')[:,:,:IMG_CHANNELS]
  img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
  X_train[n] = img
  im_gray = imread(ANNOT_PATH+'/'+str(id_)+".png",as_gray=1)
  im_gray = resize(im_gray, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
  Y_train[n] = np.expand_dims(im_gray,axis=2)


# In[47]:


def r_loss(y_true, y_pred):
    return K.mean(K.square(y_true - y_pred), axis=[1, 2, 3])


# # VAE ENCODER

# In[48]:


# ENCODER
def build_vae_encoder(input_dim, output_dim, conv_filters, conv_kernel_size,
                      conv_strides, use_batch_norm=False, use_dropout=False):
    # Clear tensorflow session to reset layer index numbers to 0 for LeakyRelu,
    # BatchNormalization and Dropout.
    # Otherwise, the names of above mentioned layers in the model
    # would be inconsistent
    global K
    K.clear_session()

    # Number of Conv layers
    n_layers = len(conv_filters)

    # Define model input
    encoder_input = Input(shape=input_dim, name='encoder_input')
    x = encoder_input

    # Add convolutional layers
    for i in range(n_layers):
        x = Conv2D(filters=conv_filters[i],
                   kernel_size=conv_kernel_size[i],
                   strides=conv_strides[i],
                   padding='same',
                   name='encoder_conv_' + str(i)
                   )(x)
        if use_batch_norm:
            x = BathcNormalization()(x)

        x = LeakyReLU()(x)

        if use_dropout:
            x = Dropout(rate=0.25)(x)

    # Required for reshaping latent vector while building Decoder
    shape_before_flattening = K.int_shape(x)[1:]

    x = Flatten()(x)

    mean_mu = Dense(output_dim, name='mu')(x)
    log_var = Dense(output_dim, name='log_var')(x)

    # Defining a function for sampling
    def sampling(args):
        mean_mu, log_var = args
        epsilon = K.random_normal(shape=K.shape(mean_mu), mean=0., stddev=1.)
        return mean_mu + K.exp(log_var / 2) * epsilon

        # Using a Keras Lambda Layer to include the sampling function as a layer

    # in the model
    encoder_output = Lambda(sampling, name='encoder_output')([mean_mu, log_var])

    return encoder_input, encoder_output, mean_mu, log_var, shape_before_flattening, Model(encoder_input,
                                                                                           encoder_output)


# In[49]:


vae_encoder_input, vae_encoder_output, mean_mu, log_var, vae_shape_before_flattening, vae_encoder = build_vae_encoder(
    input_dim=INPUT_DIM,
    output_dim=Z_DIM,
    conv_filters=[32, 64, 64, 64],
    conv_kernel_size=[3, 3, 3, 3],
    conv_strides=[2, 2, 2, 2])

vae_encoder.summary()


# In[50]:


vae_shape_before_flattening


# In[51]:


# Decoder
def build_decoder(input_dim, shape_before_flattening, conv_filters, conv_kernel_size, 
                  conv_strides):

  # Number of Conv layers
  n_layers = len(conv_filters)

  # Define model input
  decoder_input = Input(shape = (input_dim,) , name = 'decoder_input')

  # To get an exact mirror image of the encoder
  x = Dense(np.prod(shape_before_flattening))(decoder_input)
  x = Reshape(shape_before_flattening)(x)

  # Add convolutional layers
  for i in range(n_layers):
      x = Conv2DTranspose(filters = conv_filters[i], 
                  kernel_size = conv_kernel_size[i],
                  strides = conv_strides[i], 
                  padding = 'same',
                  name = 'decoder_conv_' + str(i)
                  )(x)
      
      # Adding a sigmoid layer at the end to restrict the outputs 
      # between 0 and 1
      if i < n_layers - 1:
        x = LeakyReLU()(x)
      else:
        x = Activation('sigmoid')(x)
  decoder_output = x

  return decoder_input, decoder_output, Model(decoder_input, decoder_output)


# In[61]:


vae_decoder_input, vae_decoder_output, vae_decoder = build_decoder(input_dim=Z_DIM,
                                                                   shape_before_flattening=vae_shape_before_flattening,
                                                                   conv_filters=[64, 64, 32, 1],
                                                                   conv_kernel_size=[3, 3, 3, 3],
                                                                   conv_strides=[2, 2, 2, 2])
vae_decoder.summary()


# In[62]:


vae_input = vae_encoder_input
vae_output = vae_decoder(vae_encoder_output)
vae_model = Model(vae_input, vae_output)


# In[63]:


vae_model.summary()


# In[64]:


LEARNING_RATE = 0.0005
N_EPOCHS = 200
LOSS_FACTOR = 10000


# In[65]:


get_ipython().run_line_magic('env', 'SM_FRAMEWORK=tf.keras')


# In[66]:


from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score
def kl_loss(y_true, y_pred):
    kl_loss = -0.5 * K.sum(1 + log_var - K.square(mean_mu) - K.exp(log_var), axis=1)
    return kl_loss


def total_loss(y_true, y_pred):
    return LOSS_FACTOR * r_loss(y_true, y_pred) + kl_loss(y_true, y_pred)


adam_optimizer = Adam(lr=LEARNING_RATE)

#vae_model.compile(optimizer=adam_optimizer, loss=total_loss, metrics=[r_loss, kl_loss])

vae_model.compile('Adam', loss=bce_jaccard_loss, metrics=[iou_score])


# In[67]:


checkpoint_vae = ModelCheckpoint(os.path.join(WEIGHTS_FOLDER, 'VAE/model.hdf5'), save_weights_only=True, verbose=1)


# In[68]:


X_train.shape


# In[80]:


results = vae_model.fit(X_train, Y_train, validation_split=0.1, batch_size=32, epochs=5, callbacks=[checkpoint_vae])
vae_model.save("Simple_VAE.hdf5")


# In[81]:


preds_train = vae_model.predict(X_train[:5], verbose=1)


# In[83]:


print(len(preds_train))
imshow(preds_train[0])


# In[ ]:




