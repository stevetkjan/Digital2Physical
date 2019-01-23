import numpy as np
import cv2
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
import random
import tempfile
from urllib.request import urlretrieve
from matplotlib import colors as mcolors
import matplotlib
import matplotlib.pyplot as plt
import json
import matplotlib.pyplot as plt
import os, os.path
import PIL
import tarfile




demo_steps=10
img_path='./simulated.JPEG'
output_path = './demo.jpg'
imagenet_json_path='./imagenet.json'
demo_epsilon = 40/255.0 # 
demo_lr = 0.5
demo_target = 21   ## adversarial class
img_class = 145  # original class



tf.logging.set_verbosity(tf.logging.ERROR)
sess = tf.InteractiveSession()
image = tf.Variable(tf.zeros((299, 299, 3)))
def inception(image, reuse):
    if len(image.get_shape()) == 3:
        image = tf.expand_dims(image, 0)
    preprocessed = tf.multiply(tf.subtract(image, 0.5), 2.0)
    arg_scope = nets.inception.inception_v3_arg_scope(weight_decay=0.0)
    with slim.arg_scope(arg_scope):
        logits, _ = nets.inception.inception_v3(
            preprocessed, 1001, is_training=False, reuse=reuse)
        logits = logits[:,1:] # ignore background class
        probs = tf.nn.softmax(logits) # probabilities
    return logits, probs


logits, probs = inception(image, reuse=None)





data_dir = './model'
if not os.path.isfile(os.path.join(data_dir, 'inception_v3.ckpt')):
    inception_tarball, _ = urlretrieve(
        'http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz')
    tarfile.open(inception_tarball, 'r:gz').extractall(data_dir)

restore_vars = [
    var for var in tf.global_variables()
    if var.name.startswith('InceptionV3/')
]
saver = tf.train.Saver(restore_vars)
saver.restore(sess, os.path.join(data_dir, 'inception_v3.ckpt'))


#imagenet_json, _ = urlretrieve(
#    'http://www.anishathalye.com/media/2017/07/25/imagenet.json', 'imagenet.json')
with open(imagenet_json_path) as f:
    imagenet_labels = json.load(f)





def classify(img, correct_class=None, target_class=None):
    p = sess.run(probs, feed_dict={image: img})[0]

    return [p[correct_class],p[target_class]]

def classify_output(img, correct_class=None, target_class=None):
    p = sess.run(probs, feed_dict={image: img})[0]
    print(imagenet_labels[correct_class],': ',p[correct_class] ,';' ,imagenet_labels[target_class],': ', p[target_class])
    

label = []
def save_images(images, filenames, output_dir):
    filenames = filenames
    f=output_dir+'/'+filenames

    cv2.imwrite(f, cv2.cvtColor(images*255.0, cv2.COLOR_RGB2BGR))






####Adv
x = tf.placeholder(tf.float32, (299, 299, 3))
x_hat = image # our trainable adversarial input
assign_op = tf.assign(x_hat, x)
learning_rate = tf.placeholder(tf.float32, ())
y_hat = tf.placeholder(tf.int32, ())

labels = tf.one_hot(y_hat, 1000)
loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=[labels])

epsilon = tf.placeholder(tf.float32, ())

below = x - epsilon
above = x + epsilon
projected = tf.clip_by_value(tf.clip_by_value(x_hat, below, above), 0, 1)
with tf.control_dependencies([projected]):
    project_step = tf.assign(x_hat, projected)










from math import pi
#adv=adv2
def tf_get_affine_matrix(rad=0.0, tx=0.0, ty=0.0, scale=1.0, a=0.0, b=0.0, g=0.0, h=0.0):
    # inverse
    scale = 1.0 / scale
    a = -a
    b = -b
    tx = -tx
    ty = -ty
    
    tx += 299*(1-scale)/2
    ty += 299*(1-scale)/2 
    M = tf.convert_to_tensor([scale*tf.cos(rad) - b*scale*tf.sin(rad), -scale*tf.sin(rad) + a*scale*tf.cos(rad), tx,
                              scale*tf.sin(rad) + b*scale*tf.cos(rad), scale*tf.cos(rad) + a*scale*tf.sin(rad), ty,
                              g, h])
    return M





num_samples = 10
average_loss = 0

transformed = []
for i in range(num_samples):
    rad = tf.random_uniform((), minval=-np.pi/4, maxval=np.pi/4)
    scale = tf.random_uniform((), minval=0.5, maxval=2.0)
    a = tf.random_uniform((), minval=-0.2, maxval=0.2)
    b = tf.random_uniform((), minval=-0.2, maxval=0.2)
    g = tf.random_uniform((), minval=-0.0015, maxval=0.0010)
    h = tf.random_uniform((), minval=-0.0015, maxval=0.0010)
    tx = tf.random_uniform((), minval=-30, maxval=30)
    ty = tf.random_uniform((), minval=-30, maxval=30)

    # Let tf.contrib.image.rotate perform rotate because it rotates around the center of image

    
    affine_matrix = tf_get_affine_matrix(scale=scale, a=a, b=b, g=g, h=h, tx=tx, ty=ty, rad=0.0)
    transformed.append(
        tf.contrib.image.transform(
            tf.contrib.image.rotate(image, rad), 
            affine_matrix
        )
    )
    
#print(transformed)
    
transformed_logits, _ = inception(tf.convert_to_tensor(transformed), reuse=True)







img = PIL.Image.open(img_path)
img = img.resize((299,299))
img = (np.asarray(img) / 255.0).astype(np.float32)


assign_op = tf.assign(x_hat, x)



sess.run(assign_op, feed_dict={x: img})
average_loss = 0
labels = tf.one_hot([demo_target]*num_samples, 1000)
average_loss += tf.reduce_sum(
    tf.nn.softmax_cross_entropy_with_logits(logits=transformed_logits, labels=labels)) / num_samples
optim_step = tf.train.GradientDescentOptimizer(
    learning_rate).minimize(average_loss, var_list=[x_hat])


# Target 1, most like

print('Before adding perturbations: ')
classify_output(img, correct_class=img_class, target_class=demo_target)
print('--------------')
for i in range(demo_steps):
    _, loss_value = sess.run(
        [optim_step, average_loss],
        feed_dict={learning_rate: demo_lr, y_hat: demo_target})
    sess.run(project_step, feed_dict={x: img, epsilon: demo_epsilon})
adv1 = x_hat.eval()
   # classify(adv1, correct_class=img_class, target_class=demo_target)

save_images(adv1, output_path, '.')
print('After adding perturbations: ')
classify_output(adv1, correct_class=img_class, target_class=demo_target)


