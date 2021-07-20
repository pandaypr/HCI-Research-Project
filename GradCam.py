#Imports
import sys
import os
import numpy as np
import cv2
# from matplotlib import pyplot as plt
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape, Lambda, Activation, \
    BatchNormalization, LeakyReLU, Dropout
from tensorflow.keras.models import Model
from tensorflow import keras
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from tqdm import tqdm
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
from tensorflow.python.framework import ops
import matplotlib.cm as cm
import random

#Weights
WEIGHTS_FOLDER = 'weights/'
if not os.path.exists(WEIGHTS_FOLDER):
    #  os.makedirs(os.path.join(WEIGHTS_FOLDER,"AE"))
    os.makedirs(os.path.join(WEIGHTS_FOLDER, "VAE"))

#Hyper Parameters
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3
INPUT_DIM = (IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)
BATCH_SIZE = 64
Z_DIM = 15
PATH = 'train/image'
ANNOT_PATH = 'train/label'

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


vae_encoder_input, vae_encoder_output, mean_mu, log_var, vae_shape_before_flattening, vae_encoder = build_vae_encoder(
    input_dim=INPUT_DIM,
    output_dim=Z_DIM,
    conv_filters=[32, 64, 64, 64],
    conv_kernel_size=[3, 3, 3, 3],
    conv_strides=[2, 2, 2, 2])


# vae_encoder.summary()

# Decoder
def build_decoder(input_dim, shape_before_flattening, conv_filters, conv_kernel_size,
                  conv_strides):
    # Number of Conv layers
    n_layers = len(conv_filters)

    # Define model input
    decoder_input = Input(shape=(input_dim,), name='decoder_input')

    # To get an exact mirror image of the encoder
    x = Dense(np.prod(shape_before_flattening))(decoder_input)
    x = Reshape(shape_before_flattening)(x)

    # Add convolutional layers
    for i in range(n_layers):
        x = Conv2DTranspose(filters=conv_filters[i],
                            kernel_size=conv_kernel_size[i],
                            strides=conv_strides[i],
                            padding='same',
                            name='decoder_conv_' + str(i)
                            )(x)

        # Adding a sigmoid layer at the end to restrict the outputs
        # between 0 and 1
        if i < n_layers - 1:
            x = LeakyReLU()(x)
        else:
            x = Activation('sigmoid')(x)
    decoder_output = x

    return decoder_input, decoder_output, Model(decoder_input, decoder_output)


vae_decoder_input, vae_decoder_output, vae_decoder = build_decoder(input_dim=Z_DIM,
                                                                   shape_before_flattening=vae_shape_before_flattening,
                                                                   conv_filters=[64, 64, 64, 1],
                                                                   conv_kernel_size=[3, 3, 3, 3],
                                                                   conv_strides=[2, 2, 2, 2])
# vae_decoder.summary()

#Custom Data Generator using yield instead of Return, very efficient and customizable


def data_gen(img_folder, mask_folder, batch_size):
    c = 0
    n = list(int(s.split(".")[0]) for s in list(next(os.walk(img_folder))[2]))
    random.shuffle(n)

    while (True):
        img = np.zeros((batch_size, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)).astype('float')
        mask = np.zeros((batch_size, IMG_HEIGHT, IMG_WIDTH, 1)).astype('float')

        for i in range(c, c + batch_size):  # initially from 0 to 16, c = 0.

            train_img = cv2.imread(img_folder + '/' + str(n[i]) + '.jpg') / 255.
            train_img = cv2.resize(train_img, (IMG_HEIGHT, IMG_WIDTH))  # Read an image from folder and resize

            img[i - c] = train_img  # add to array - img[0], img[1], and so on.

            train_mask = cv2.imread(mask_folder + '/' + str(n[i]) + '.png', cv2.IMREAD_GRAYSCALE) / 255.
            train_mask = cv2.resize(train_mask, (IMG_HEIGHT, IMG_WIDTH))
            train_mask = train_mask.reshape(IMG_HEIGHT, IMG_WIDTH,
                                            1)  # Add extra dimension for parity with train_img size [512 * 512 * 3]

            mask[i - c] = train_mask

        c += batch_size
        if (c + batch_size >= len(os.listdir(img_folder))):
            c = 0
            random.shuffle(n)

        yield img, mask


train_frame_path = 'train/image'
train_mask_path = 'train/label'

val_frame_path = 'Dataset/images/train images'
val_mask_path = 'Dataset/masks/train label'

# Train the model
train_gen = data_gen(train_frame_path, train_mask_path, batch_size=BATCH_SIZE)
val_gen = data_gen(val_frame_path, val_mask_path, batch_size=BATCH_SIZE)


#A Function to load images from one folder used in testing
def load_image(path, preprocess=True):
    """Load and preprocess image."""
    train_ids = list(int(s.split(".")[0]) for s in list(next(os.walk(path))[2]))
    image = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    sys.stdout.flush()
    for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
        img = imread(path + '/' + str(id_) + '.jpg')[:, :, :IMG_CHANNELS]
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        image[n] = img.astype('uint8') / 255
    return image

#reminiscent of previous guided Grad-CAM codes. not used here, but can be helpful for future works

'''def build_guided_model():
    """Function returning modified model.

    Changes gradient function for all ReLu activations
    according to Guided Backpropagation.
    """
    if "GuidedBackProp" not in ops._gradient_registry._registry:
        @ops.RegisterGradient("GuidedBackProp")
        def _GuidedBackProp(op, grad):
            dtype = op.inputs[0].dtype
            return grad * tf.cast(grad > 0., dtype) * \
                   tf.cast(op.inputs[0] > 0., dtype)

    g=tf.compat.v1.get_default_graph()     
    with g.gradient_override_map({'Relu': 'GuidedBackProp'}):
        new_model = build_model()
        new_model.summary()
    return new_model'''

#Build VAE Model
def build_model():
    vae_input = vae_encoder_input
    vae_output = vae_decoder(vae_encoder_output)
    vae_model = Model(vae_input, vae_output)

    return vae_model

#Losses::: Since it is a segmentation task, Jaccard indea and IOU score is used for calculating the loss in training

from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score
from tensorflow.keras.optimizers import Adam

LEARNING_RATE = 0.0005
N_EPOCHS = 4
LOSS_FACTOR = 10000
SM_FRAMEWORK = tf.keras
model = build_model()
model.compile('Adam', loss=bce_jaccard_loss, metrics=[iou_score])

#checkpoint to save the model weights
checkpoint_vae = ModelCheckpoint(os.path.join(WEIGHTS_FOLDER, 'VAE/SegGradCAM.hdf5'), save_weights_only=True, verbose=1)


NO_OF_TRAINING_IMAGES = len(os.listdir(train_frame_path))
NO_OF_VAL_IMAGES = len(os.listdir(val_frame_path))

#Training the model
model.fit(train_gen, epochs=N_EPOCHS,
          steps_per_epoch=(NO_OF_TRAINING_IMAGES // BATCH_SIZE),
          validation_data=val_gen,
          validation_steps=(NO_OF_VAL_IMAGES // BATCH_SIZE),
          # callbacks=callbacks_list,
          callbacks=[checkpoint_vae])


#Function to make Grad-CAM heatmap

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, encoder_out, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.get_layer(encoder_out).output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        # preds = preds.squeeze()
        # class_channel = preds[2][0][0]
    # print("shape of class_channel: "+str(tf.shape(class_channel)))
    # print("class_channel: "+str((class_channel)))
    # print("last_conv_layer_output: "+str(last_conv_layer_output))

    grads = tape.gradient(preds, last_conv_layer_output)

    """with tf.GradientTape() as tape_1:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[2])
        class_channel = preds[2][0][1]
    grads_1 = tape_1.gradient(class_channel, last_conv_layer_output)"""

    # print("grads: "+str((grads)))
    # print("###############grads_1: "+str((grads_1)))
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    # pooled_grads_1 = tf.reduce_mean(grads_1, axis=(0, 1, 2))
    # print("pooled_grads: "+str(tf.shape(pooled_grads)))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    # heatmap_1 = last_conv_layer_output @ pooled_grads_1[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    # heatmap_1 = tf.squeeze(heatmap_1)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    # heatmap_1 = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap_1)
    # print("heatmap: "+str((heatmap)))
    return heatmap.numpy()

#Fetching indicidual Layers from the VAE Model
last_conv_layer_name = "encoder_conv_3"
encoder_out = "encoder_output"
#Load image to generate heatmaps
img_path = 'Dataset/images/test images'
a = load_image(img_path)
img_array = a[0]
img_array = keras.preprocessing.image.img_to_array(img_array)
img_array = np.expand_dims(img_array, axis=0)

#Predict the image mask output
res = model.predict(img_array)

# Call the heatmap function
heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name, encoder_out)

#To display and save the VAE output Mask image
res = res.squeeze()
from PIL import Image
res = res * 255
res = res.astype(np.uint8)
res = Image.fromarray(res)
# imshow(res)
res.save("VAE_generated_image.jpeg")

#To display and save the Heatmap of the last Convulution Layer
hm = np.uint8(255 * heatmap)
hm = np.expand_dims(hm, axis=2)
hm = keras.preprocessing.image.array_to_img(hm)
print(img_array.shape)
hm = hm.resize((img_array.shape[1], img_array.shape[2]))
hm = np.asanyarray(hm)
imshow(hm)
# hm.save("heatmap.jpg")
cv2.imwrite('hm.jpeg', hm)

#Superimpose heatmap on the actual image
def save_and_display_gradcam(img, heatmap, cam_path="superimposed_img.jpg", alpha=0.4):
    # img = keras.preprocessing.image.load_img(img_path)
    # img = keras.preprocessing.image.img_to_array(img)

    heatmap = np.uint8(255 * heatmap)
    # heatmap_1 = np.uint8(255 * heatmap_1)
    # hmap = hmap.resize((img.shape[1], img.shape[0]))
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)
    print(jet_heatmap.shape)
    print(img.shape)
    superimposed_img = cv2.addWeighted(jet_heatmap, 0.005, img, 0.995, 0)
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)
    superimposed_img.save(cam_path)


save_and_display_gradcam(img_array.squeeze(), heatmap)

#To display heatmap for individual Z-space Dimensions
heatmap_images = 'opop'


def save_individual_heatmaps_1(img_array, model, last_conv_layer_name, encoder_out, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.get_layer(encoder_out).output]
    )
    no_of_channels = (model.get_layer(last_conv_layer_name).output.get_shape())[3]  # change 1
    print(no_of_channels)
    for i in range(no_of_channels):
        with tf.GradientTape(persistent=True) as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            y = last_conv_layer_output[:, :, :, i]  # change 2
            if (i == 1909):
                y = tf.math.multiply(y, tf.constant(1000, dtype=tf.float32))
        grads = tape.gradient(y, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = tf.matmul(last_conv_layer_output, pooled_grads[..., tf.newaxis])
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        # if(i==199):
        # print("heatmap: "+str(heatmap))
        # print("mean: "+str(tf.math.reduce_mean(heatmap)))
        # save_and_display_gradcam(img_array.squeeze(), heatmap)
        save_and_display_gradcam(img_array.squeeze(), heatmap, cam_path=heatmap_images + "/heatmap_" + str(i) + ".jpg")

import tensorflow.python.ops.numpy_ops.np_config as np_config
np_config.enable_numpy_behavior()

def save_individual_heatmaps_(img_array, model, last_conv_layer_name, encoder_out, pred_index=None):
     grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.get_layer(encoder_out).output]
    )
     last_conv_layer_output, preds = grad_model(img_array)
     a=last_conv_layer_output[:,:,:,1]
     print(a)
     a=a*255
     a=tf.dtypes.cast(a, tf.int32)
     a=a.numpy()
     #a.squeeze(a)
     #res = a*255
     res = a.astype(np.uint8)
     res = Image.fromarray(res)
     imshow(res)

save_individual_heatmaps_(img_array, model, last_conv_layer_name, encoder_out)

