# Example on how to use the tensorflow input pipelines. The explanation can be found here ischlag.github.io.
import tensorflow as tf
import random
import pandas as pd
import numpy as np
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import os.path

dataset_path      = "train/"
train_labels_file = "train_master.tsv"
test_path = "test/"
test_laabels_file = "sample_submit.csv"
logdir = '/tmp/uniqlo_2/'

IMAGE_HEIGHT  = 133
IMAGE_WIDTH   = 133
NUM_CHANNELS  = 3
BATCH_SIZE = 100
TEST_BATCH_SIZE = 100

def encode_label(label):
  return int(label)

def read_label_file(file):
  filepaths = []
  labels = []
  f = pd.read_csv(file,delimiter='\t')
  filepaths = list(f.iloc[:,0])
  labels = list(f.iloc[:,1])
  return filepaths, labels

def file_loader():
    
    # reading labels and file path
    train_filepaths, train_labels = read_label_file(train_labels_file)
    num_files = len(train_filepaths)
    
    # transform relative path into full path
    train_filepaths = [ dataset_path + fp for fp in train_filepaths]
    
    # convert string into tensors
    train_filepaths = ops.convert_to_tensor(train_filepaths, dtype=dtypes.string)
    train_labels = ops.convert_to_tensor(train_labels, dtype=dtypes.int32)
    
    if os.path.exists(logdir+"partition.csv"):
        partitions = np.array(pd.read_csv(logdir+"partition.csv",header=None))
        partitions = list(partitions.ravel())
    else:
        # create a partition vector
        partitions = [0] * num_files
        partitions[:int(0.3*num_files)] = [1] * int(0.3*num_files)
        random.shuffle(partitions)
        np.savetxt(logdir+"/partition.csv", partitions, delimiter=",", fmt='%s')
    
    
    # partition our data into a test and train set according to our partition vector
    train_images, eval_images = tf.dynamic_partition(train_filepaths, partitions, 2)
    train_labels, eval_labels = tf.dynamic_partition(train_labels, partitions, 2)
    
    return train_images, train_labels, eval_images, eval_labels

def input():
    train_images, train_labels, _, _ = file_loader()
    train_input_queue = tf.train.slice_input_producer(
                                    [train_images, train_labels],
                                    shuffle=True)

    
    # process path and string tensor into an image and a label
    file_content = tf.read_file(train_input_queue[0])
    train_image = tf.image.decode_jpeg(file_content, channels=NUM_CHANNELS)
    train_image = tf.image.resize_images(train_image, [IMAGE_HEIGHT, IMAGE_WIDTH])
    train_image = tf.image.per_image_standardization(train_image)
    #train_image = tf.cast(train_image, tf.float32)
    train_label = train_input_queue[1]
    

    
    # define tensor shape
    train_image.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS])
    
    
    # collect batches of images before processing
    train_image_batch, train_label_batch = tf.train.batch(
                                    [train_image, train_label],
                                    batch_size=BATCH_SIZE
                                    #,num_threads=1
                                    )
    
    
    return train_image_batch, train_label_batch

def input_eval():
    _, _, eval_images, eval_labels = file_loader()
    eval_input_queue = tf.train.slice_input_producer(
                                    [eval_images, eval_labels],
                                    shuffle=True)
    
    file_content = tf.read_file(eval_input_queue[0])
    eval_image = tf.image.decode_jpeg(file_content, channels=NUM_CHANNELS)
    eval_image = tf.image.resize_images(eval_image, [IMAGE_HEIGHT, IMAGE_WIDTH])
    eval_image = tf.image.per_image_standardization(eval_image)

    eval_label = eval_input_queue[1]
    
    eval_image.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS])
    
    eval_image_batch, eval_label_batch = tf.train.batch(
                                    [eval_image, eval_label],
                                    batch_size=BATCH_SIZE
                                    #,num_threads=1
                                    )
  
    return eval_image_batch, eval_label_batch

def input_test():
    # reading labels and file path
    f = pd.read_csv(test_laabels_file, header=None)
    test_filepaths = list(f[0])
    test_labels = list(f[1])
    num_files = len(test_filepaths)
    
    # transform relative path into full path
    test_filepaths = [ test_path + fp for fp in test_filepaths]
    
    # convert string into tensors
    test_filepaths = ops.convert_to_tensor(test_filepaths, dtype=dtypes.string)
    test_labels = ops.convert_to_tensor(test_labels, dtype=dtypes.int32)
    
    test_input_queue = tf.train.slice_input_producer(
                                    [test_filepaths, test_labels],
                                    shuffle=False)
    file_content = tf.read_file(test_input_queue[0])
    test_image = tf.image.decode_jpeg(file_content, channels=NUM_CHANNELS)
    test_image = tf.image.resize_images(test_image, [IMAGE_HEIGHT, IMAGE_WIDTH])
    test_image = tf.image.per_image_standardization(test_image)

    test_label = test_input_queue[1]
    
    test_image.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS])
    
    test_image_batch, test_label_batch = tf.train.batch(
                                    [test_image, test_label],
                                    batch_size=TEST_BATCH_SIZE
                                    #,num_threads=1
                                    )
  
    return test_image_batch, test_label_batch, num_files

###################################################################################
'''

img, label = input()
with tf.Session() as sess:
  
  # initialize the variables
  sess.run(tf.global_variables_initializer())
  
  # initialize the queue threads to start to shovel data
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)

  print("from the train set:")
  #img = sess.run(img[1])/5
  #io.imshow(img)
  print(sess.run(label))
  #example = sess.run(test_image_batch[2])/255
  #io.imshow(example)
  
  # stop our queue threads and properly close the session
  coord.request_stop()
  coord.join(threads)
  sess.close()

'''

