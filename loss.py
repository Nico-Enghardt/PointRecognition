import tensorflow as tf
import numpy as np

def loss3D(y_true,y_pred):
    # Plane Error + Height Error * 3
    return tf.sqrt(tf.reduce_sum(tf.reduce_mean((y_true[:,0:2] - y_pred[:,0:2])**2, axis=0))) + 3*tf.sqrt(tf.reduce_mean(((y_true[:,2] - y_pred[:,2])**2)))
    
def planeError(y_true,y_pred):
    # Difference of X- and Y-Dimension, square each value, average over each dimension, add both dimension values and pull pythagorean root
    return tf.sqrt(tf.reduce_sum(tf.reduce_mean((y_true[:,0:2] - y_pred[:,0:2])**2, axis=0)))

def heightError(y_true,y_pred):
    # Difference of height dimension, square each value, then average all differences and squareroot the result 
    return tf.sqrt(tf.reduce_mean(((y_true[:,2] - y_pred[:,2])**2)))
