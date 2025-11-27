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
        self._replay_suffix = replay_suffix # which replay buffer to load
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
            self._num_replay_buffers = 1
            self._loaded_buffers = True


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
    
    def _load_replay_buffers(self, num_buffers=None):
        '''Loads several checkpoints into a list of OutOfGraphReplayBuffers'''
        if not self._loaded_buffers:
            ckpts = gfile.ListDirectory(self._data_dir)
            # FIXME: The author assumes that the checkpoints are saved in a format CKPT_NAME.{SUFFIX}.gz
            ckpt_counters = collections.Counter(
                [name.split('.')[-2] for name in ckpts])
            # Should contain the files for add_count, action, observation, reward, terminal and invalid_range
            ckpt_suffixes = [x for x in ckpt_counters if ckpt_counters[x] in [6, 7]]
            if num_buffers is not None:
                ckpt_suffixes = np.random.choice(
                    ckpt_suffixes, num_buffers, replace=False)
            self._replay_buffers = []
            # Load the replay buffers in parallel
            with futures.ThreadPoolExecutor(
                max_workers=num_buffers) as thread_pool_executor:
                replay_futures = [thread_pool_executor.submit(
                    self._load_buffer, suffix) for suffix in ckpt_suffixes]
            for f in replay_futures:
                replay_buffer = f.result()
                if replay_buffer is not None:
                self._replay_buffers.append(replay_buffer)
                self.add_count = max(replay_buffer.add_count, self.add_count)
            self._num_replay_buffers = len(self._replay_buffers)
            if self._num_replay_buffers:
                self._loaded_buffers = True

    def get_transition_elements(self):
        return self._replay_buffers[0].get_transition_elements()

    def sample_transition_batch(self, batch_size=None, indices=None):
        buffer_index = np.random.randint(self._num_replay_buffers)
        return self._replay_buffers[buffer_index].samople_transition_batch(
            batch_size=batch_size, indices=indices
        )
    
    def load(self, *args, **kwargs):
        pass

    def reload_buffer(self, num_buffers=None):
        self._loaded_buffers = False
        self._load_replay_buffers(num_buffers)

    def save(self, *args, **kwargs):
        pass

    def add(self, *args, **kwargs):
        pass