from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time

import tensorflow as tf
import numpy as np

import model

FLAGS = tf.app.flags.FLAGS
batch_size = model.batch_size
max_steps = 1000000
#tf.app.flags.DEFINE_integer('max_steps', 1000000, """Number of batches to run.""")
log_device_placement = False
#tf.app.flags.DEFINE_boolean('log_device_placement', True, """Whether to log device placement.""")
log_frequency = 10
#tf.app.flags.DEFINE_integer('log_frequency', 10, """How often to log results to the console.""")
logdir='/tmp/uniqlo_1'

def train():
  """Train CIFAR-10 for a number of steps."""
  with tf.Graph().as_default():
        global_step = tf.contrib.framework.get_or_create_global_step()
        keep_prob3 = tf.placeholder(tf.float32)
        keep_prob4 = tf.placeholder(tf.float32)
    
        # Get images and labels for CIFAR-10.
        # Force input pipeline to CPU:0 to avoid operations sometimes ending up on
        # GPU and resulting in a slow down.
        with tf.device('/cpu:0'):
          images, labels = model.input()
          #images_eval, labels_eval = model.input_eval()
    
        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = model.inference(images, keep_prob3=0.7, keep_prob4=0.5)
    
        # Calculate loss.
        loss = model.loss(logits, labels)
        
    
        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        train_op = model.train(loss, global_step)
        
        tf.summary.scalar("cost_value_loss", loss) 
        summary_op = tf.summary.merge_all()
        
        class _LoggerHook(tf.train.SessionRunHook):
          """Logs loss and runtime."""
    
          def begin(self):
            self._step = -1
            self._start_time = time.time()
            
          def before_run(self, run_context):
            self._step += 1
            
            return tf.train.SessionRunArgs(loss)  # Asks for loss value.
    
          def after_run(self, run_context, run_values):
            if self._step % log_frequency == 0:
              current_time = time.time()
              duration = current_time - self._start_time
              self._start_time = current_time
              
              loss_value = run_values.results
              examples_per_sec = log_frequency * batch_size / duration
              sec_per_batch = float(duration / log_frequency)
    
              format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                            'sec/batch)')
              print (format_str % (datetime.now(), self._step, loss_value,
                                   examples_per_sec, sec_per_batch))
    
           
        with tf.train.MonitoredTrainingSession(
            checkpoint_dir=logdir,
            hooks=[tf.train.StopAtStepHook(last_step=max_steps),
                   tf.train.NanTensorHook(loss),
                   _LoggerHook(),
                   tf.train.SummarySaverHook(save_steps=log_frequency,output_dir=logdir,summary_op=summary_op)],
            config=tf.ConfigProto(log_device_placement=log_device_placement)) as mon_sess:
            
            while not mon_sess.should_stop():
                
                #summary_writer = tf.summary.FileWriter('/tmp/uniqlo_1',mon_sess.graph)
                mon_sess.run(train_op)
                
                



def main(argv=None):  # pylint: disable=unused-argument
  train()


if __name__ == '__main__':
  tf.app.run()