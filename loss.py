import tensorflow as tf
import numpy as np

a = tf.constant([[0,2,5],[0,2,4],[4,3,9],[3,5,5]],dtype=float)
b = tf.constant([[3,4,2],[0,5,6],[0,2,7],[2,3,5]],dtype=float)

def loss3D(y_true,y_pred):
    # Plane Error + Height Error * 5
    return tf.sqrt(tf.reduce_sum(tf.reduce_mean((y_true[:,0:2] - y_pred[:,0:2])**2, axis=0))) + 5*tf.sqrt(tf.reduce_mean(((y_true[:,2] - y_pred[:,2])**2)))
    
def planeError(y_true,y_pred):
    # Difference of X- and Y-Dimension, square each value, average over each dimension, add both dimension values and pull pythagorean root
    return tf.sqrt(tf.reduce_sum(tf.reduce_mean((y_true[:,0:2] - y_pred[:,0:2])**2, axis=0)))

def heightError(y_true,y_pred):
    # Difference of height dimension, square each value, then average all differences and squareroot the result 
    return tf.sqrt(tf.reduce_mean(((y_true[:,2] - y_pred[:,2])**2)))

print(heightError(a,b))
print(planeError(a,b))
print(loss3D(a,b))

print(heightError(b,a))
print(planeError(b,a))
print(loss3D(b,a))
