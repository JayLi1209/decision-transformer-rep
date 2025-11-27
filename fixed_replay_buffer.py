import collections
from concurrent import futures
from dopamine.replay_memory import circular_replay_buffer
import numpy as np
import tensorflow.compat.v1 as tf
import gin

gfile = tf.gfile

STORE_FILENAME_PREFIX = circular_replay_buffer.STORE_FILENAME_PREFIX

class FixedReplayBuffer(object):
    '''A list of OutofGraphReplayBuffers'''

    def __init__(self, data_dir, replay_suffix, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs
        self._data_dir = data_dir
        self._loaded_buffers = False
        self.add_count = np.array(0)
        self._replay_suffix = replay_suffix
        if not self._loaded_buffers:
            if replay_suffix is not None:
                assert replay_suffix >= 0, 'Please pass a non-negative replay suffix!'
                self.load_single_buffer(replay_suffix)
            else:
                self._load_replay_buffers(num_buffers=50)
    
    def load_single_buffer(self, suffix):
        replay_buffer = self._load_buffer(suffix)
        if replay_buffer is not None:
            self._replay_buffers = [replay_buffer]
            self.add_count = replay_buffer.add_count

    










    def _load_buffer(self, suffix):
        '''Loads a OutOfGraphReplayBuffer replay buffer!'''
        try:
            replay_buffer = circular_replay_buffer.OutOfGraphReplayBuffer(
                *self._args, **self._kwargs
            )
            replay_buffer.load(self._data_dir, suffix)
            tf.logging.info('Loaded replay buffer ckpt {} from {}'.format(
                suffix, self._data_dir
            ))
            return replay_buffer
        except tf.errors.NotFoundError:
            return None
    
