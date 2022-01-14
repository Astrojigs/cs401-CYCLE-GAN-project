import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *



# LOSSES
LAMBDA = 15
loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real, generated):
    real_loss = loss_obj(tf.ones_like(real), real)

    generated_loss = loss_obj(tf.zeros_like(generated), generated)

    total_disc_loss = real_loss+generated_loss

    return total_disc_loss*0.5

def generator_loss(generated):
    return loss_obj(tf.ones_like(generated), generated)

def calc_cycle_loss(real_image, cycle_image):
    loss1 = tf.reduce_mean(tf.abs(real_image-cycle_image))
    return LAMBDA*loss1

''' generator_g is responsible for translating image X to image Y. Identity
loss says that, if you fed image Y to generator G, it should yield the real
image Y or something close to image Y.
    identity_loss = |G(Y)-Y|+|F(X)-X|'''
def identity_loss(real_image, same_image):
    loss = tf.reduce_mean(tf.abs(real_image - same_image))
    print(loss)
    return LAMBDA*0.5*loss
