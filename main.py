import numpy as np
import tensorflow as tf
import cv2
import os
import time
from tensorflow.keras.layers import *
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import clear_output

# self-made python files:
from losses import *
from models import *
from utils import *

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


# Changing resolution of data images to 128x128
from PIL import Image
# # Resizing images code
# images_path = '../final project/horse2zebra/testB'
# size = 128, 128 # (horizontal pixels, vertical pixels)
# for image_name, num in zip(os.listdir(images_path), range(len(os.listdir(images_path)))):
#     full_path = os.path.join(images_path, image_name)
#     # changing resolution
#     img = Image.open(full_path)
#     resized_img = img.resize(size, Image.ANTIALIAS)
#     os.remove(images_path + '/' +image_name)
#     resized_img.save(images_path + '/'+str(num) + '.jpg')


# Paths
train_horse_path = '../final project/horse2zebra/trainA'
train_zebra_path = '../final project/horse2zebra/trainB'
train_horse = []
train_zebra = []

# For face images
for image_name in os.listdir(train_horse_path):
    full_img_name = os.path.join(train_horse_path,image_name)
    img_arr = plt.imread(full_img_name)
    if img_arr.shape != (128,128,3):
        continue
    # normalizing images from 0 to 1
    train_horse.append(img_arr/255)
# For anime images
for image_name in os.listdir(train_zebra_path):
    full_img_name = os.path.join(train_zebra_path, image_name)
    img_arr = plt.imread(full_img_name)
    if img_arr.shape != (128,128,3):
        continue
    # normalizing images from 0 to 1
    train_zebra.append(img_arr/255)

train_horse = np.array(train_horse, dtype='float32')
train_zebra = np.array(train_zebra, dtype = 'float32')
print(train_horse.shape)
print(train_zebra.shape)




IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
CHANNELS = 3


train_horses = tf.data.Dataset.from_tensor_slices(train_horse).batch(1)
train_zebras = tf.data.Dataset.from_tensor_slices(train_zebra).batch(1)


#########################################
#########################################
# Building generators and discriminators
generator_g = Generator()
generator_f = Generator()

discriminator_x = Discriminator()
discriminator_y = Discriminator()





###############################################################




# Optimizers
generator_g_optimizer = tf.keras.optimizers.Adam(2e-3, beta_1=0.5)
generator_f_optimizer = tf.keras.optimizers.Adam(2e-3, beta_1=0.5)

discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-3, beta_1=0.5)
discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-3, beta_1=0.5)




EPOCHS = 300



for epoch in range(EPOCHS):
    start = time.time()

    n = 0
    for en_image_x, en_image_y in tf.data.Dataset.zip((train_horses, train_zebras)):

        train_step(en_image_x, en_image_y)
        if n % 10 == 0:
            print ('.', end='')
        n+=1

    clear_output(wait=True)

    #Random image:
    rnd = np.random.randint(0, len(train_horse))
    random_img = train_horse[rnd]
    plt.imshow(random_img)
    plt.title('Original Image')
    plt.show()
    generate_img(generator_g,random_img.reshape((1,)+random_img.shape[:]))
    if (epoch + 1) % 5 == 0:
        pass
        #ckpt_save_path = ckpt_manager.save()
        #print(f'Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}')

    print(f'Time taken for epoch {epoch + 1} is {time.time() - start}')
