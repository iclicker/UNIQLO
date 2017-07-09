# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Evaluation for CIFAR-10.

Accuracy:
cifar10_train.py achieves 83.0% accuracy after 100K steps (256 epochs
of data) as judged by cifar10_eval.py.

Speed:
On a single Tesla K40, cifar10_train.py processes a single batch of 128 images
in 0.25-0.35 sec (i.e. 350 - 600 images /sec). The model reaches ~86%
accuracy after 100K steps in 8 hours of training time.

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import pandas as pd
import math
import time

import numpy as np
np.set_printoptions(threshold=np.inf)
import tensorflow as tf

import model
import img_loader
import train

FLAGS = tf.app.flags.FLAGS
'''
tf.app.flags.DEFINE_string('eval_dir', '/tmp/uniqlo_eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/uniqlo',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 10000,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                         """Whether to run eval only once.""")
'''

batch_size = model.batch_size
test_batch_size = img_loader.TEST_BATCH_SIZE
eval_dir = '/tmp/uniqlo_eval'
eval_data = 'test'
#checkpoint_dir = '/tmp/uniqlo_2'
checkpoint_dir = train.logdir
eval_interval_secs = 30
num_examples = 3000
run_once = False

def eval_once(saver, summary_writer, top_k_op, summary_op):
  """Run Eval once.

  Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_k_op: Top K op.
    summary_op: Summary op.
  """
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)
      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/cifar10_train/model.ckpt-0,
      # extract global_step from it.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
      print('No checkpoint file found')
      return

    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

      num_iter = int(math.ceil(num_examples / batch_size))
      true_count = 0  # Counts the number of correct predictions.
      total_sample_count = num_iter * batch_size
      step = 0
      while step < num_iter and not coord.should_stop():
        predictions = sess.run([top_k_op])
        true_count += np.sum(predictions)
        step += 1

      # Compute precision @ 1.
      precision = true_count / total_sample_count
      print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))
      print('total correct', true_count)
      print('total sample', total_sample_count)

      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
      summary.value.add(tag='Precision @ 1', simple_value=precision)
      summary_writer.add_summary(summary, global_step)
    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)


def evaluate():
  """Eval CIFAR-10 for a number of steps."""
  with tf.Graph().as_default() as g:
    # Get images and labels for CIFAR-10.
    images, labels = model.input()

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = model.inference(images)

    # Calculate predictions.
    top_k_op = tf.nn.in_top_k(logits, labels, 1)

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        model.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    summary_writer = tf.summary.FileWriter(eval_dir, g)

    while True:
      eval_once(saver, summary_writer, top_k_op, summary_op)
      if run_once:
        break
      time.sleep(eval_interval_secs)


def main(argv=None):  # pylint: disable=unused-argument
  #evaluate()
  predict()
  
def predict():
    images, labels = model.input_eval()
    logits = model.inference(images)
    logits = tf.nn.softmax(logits)
    prediction = tf.argmax(logits, 1)
    
    variable_averages = tf.train.ExponentialMovingAverage(model.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        saver.restore(sess, ckpt.model_checkpoint_path)
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        
        num_iter = int(math.ceil(num_examples / test_batch_size))
        step = 0
        result = []
        label = []
        
        while step < num_iter:
            #pred = sess.run(prediction)
            pred, y = sess.run([prediction,labels])
            result.append(pred)
            label.append(y)
            step += 1
            
        coord.request_stop()
        coord.join(threads)
        sess.close()
    
    result = [item for sublist in result for item in sublist]   
    label = [item for sublist in label for item in sublist]
    accuracy, balanced, output = ba(result, label)
    print('total accuracy', accuracy)
    print('balanced accuracy', balanced)
    return output

def ba(y, y_):
    df = pd.DataFrame([y,y_]).transpose()
    df.columns=(['prediction','label'])
    ba=pd.DataFrame()
    total_correct = len(df.loc[df.prediction==df.label].index)
    accuracy = total_correct / len(df)
    balanced = 0
    
    ba['cat']=list(range(0,24))
    ba['cat_total']=0
    ba['cat_correct']=0
    for i in ba.cat:
        ba.cat_total[i] = len(df.loc[df.label==i].index)
        ba.cat_correct[i] = len(df.loc[(df.prediction==df.label)&(df.label==i)].index)
        balanced += ba.cat_correct[i] / ba.cat_total[i]
    balanced = balanced / 24
    
    return accuracy, balanced, ba
        

def write(file, result):
    test_laabels_file = "sample_submit.csv"
    f = pd.read_csv(test_laabels_file, header=None)
    f.iloc[:,1] = result
    f.to_csv(test_laabels_file, header=False, index=False)

'''
if __name__ == '__main__':
  tf.app.run()
  
'''