# @title Necessary imports and globals.
from dopamine.replay_memory import prioritized_replay_buffer
import collections
import tensorflow as tf
import numpy as np
import os
import dopamine
from dopamine.agents.dqn import dqn_agent
from dopamine.discrete_domains import run_experiment, atari_lib
from dopamine.colab import utils as colab_utils
from absl import flags
import gin.tf

BASE_PATH = 'running-data'  # @param
GAME = 'Berzerk'  # @param
LOG_PATH = os.path.join(BASE_PATH, 'prioritized_srdqn_normalized', GAME)

SRNetworkType = collections.namedtuple(
    'sr_network', ['feature', 'decoded_state', 'sr_values'])


class SRNetwork(tf.keras.Model):
  """The convolutional network used to compute the agent's successor features"""

  def __init__(self, num_actions, in_channnels, name=None):
    """Creates the layers used for calculating Q-values.

    Args:
      num_actions: int, number of actions.
      name: str, used to create scope for network parameters.
    """
    super(SRNetwork, self).__init__(name=name)

    self.num_actions = num_actions
    self.in_channels = in_channnels

    # Defining layers.
    activation_fn = tf.keras.activations.relu
    # Setting names of the layers manually to make variable names more similar
    # with tf.slim variable names/checkpoints.
    self.conv1 = tf.keras.layers.Conv2D(32, [8, 8], strides=4, padding='same',
                                        activation=activation_fn, name='Conv')
    self.conv2 = tf.keras.layers.Conv2D(64, [6, 6], strides=3, padding='same',
                                        activation=activation_fn, name='Conv')
    self.conv3 = tf.keras.layers.Conv2D(64, [3, 3], strides=1, padding='same',
                                        activation=activation_fn, name='Conv')
    self.flatten = tf.keras.layers.Flatten()
    self.dense_phi = tf.keras.layers.Dense(
        512, name='fully_connected')  # the phi_st

    # auto-decoder to reconstruct state
    self.dense_decoder = tf.keras.layers.Dense(
        7 * 7 * self.in_channels, activation=activation_fn, name='dense_decoder')
    self.reshape_decoder = tf.keras.layers.Reshape((7, 7, self.in_channels))
    self.conv4 = tf.keras.layers.Conv2DTranspose(64, [3, 3], strides=1, padding='same',
                                                 activation=activation_fn, name='Conv')
    self.conv5 = tf.keras.layers.Conv2DTranspose(64, [6, 6], strides=3, padding='same',
                                                 activation=activation_fn, name='Conv')
    self.conv6 = tf.keras.layers.Conv2DTranspose(32, [8, 8], strides=4, padding='same',
                                                 activation=activation_fn, name='Conv')
    self.conv_st = tf.keras.layers.Conv2D(self.in_channels, [4, 4], strides=1, padding='same',
                                          name='Conv')  # the output s^t

    # successor representation branches
    self.branches = [[
        tf.keras.layers.Dense(512, name='fully_connected',
                              activation=activation_fn),
        tf.keras.layers.Dense(256, name='fully_connected',
                              activation=activation_fn),
        tf.keras.layers.Dense(512, name='fully_connected')
    ] for i in range(self.num_actions)]

  def call(self, state):
    """Creates the output tensor/op given the state tensor as input.

    See https://www.tensorflow.org/api_docs/python/tf/keras/Model for more
    information on this. Note that tf.keras.Model implements `call` which is
    wrapped by `__call__` function by tf.keras.Model.

    Parameters created here will have scope according to the `name` argument
    given at `.__init__()` call.
    Args:
      state: Tensor, input tensor.
    Returns:
      collections.namedtuple, output ops (graph mode) or output tensors (eager).
    """

    # compute feature
    x = tf.cast(state, tf.float32)
    x = x / 255
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.flatten(x)
    phi = self.dense_phi(x)

    # compute decoded feature
    phi_st = self.dense_decoder(phi)
    phi_st = self.reshape_decoder(phi_st)
    phi_st = self.conv4(phi_st)
    phi_st = self.conv5(phi_st)
    phi_st = self.conv6(phi_st)
    phi_st = self.conv_st(phi_st)
    decoded_state = tf.multiply(phi_st, 255)

    # compute successor representation
    srs = []
    for i in range(self.num_actions):
      branch = self.branches[i]
      sr = branch[0](phi)
      sr = branch[1](sr)
      sr = branch[2](sr)
      srs.append(sr)
    srs = tf.convert_to_tensor(srs)

    return SRNetworkType(phi, decoded_state, srs)


# @title Create the DQN with prioritized replay


class PrioritizedSRDQNAgent(dqn_agent.DQNAgent):
  def __init__(self, sess, num_actions):
    """This maintains all the DQN default argument values."""
    super().__init__(sess, num_actions, tf_device='/gpu:0')
    self._replay_scheme = 'prioritized'

    with tf.device('/gpu:0'):
        self._sr_train_op = self._build_sr_train_op()

    print('finished constructing')
    self.online_convnet.summary()
    self.sr_convnet.summary()

  def _build_networks(self):
    """Builds the Q-value network computations needed for acting and training.

    These are:
      self.online_convnet: For computing the current state's Q-values.
      self.target_convnet: For computing the next state's target Q-values.
      self.sr_convnet: For computing the sr for state-action pair
      self._net_outputs: The actual Q-values.
      self._q_argmax: The action maximizing the current state's Q-values.
      self._replay_net_outputs: The replayed states' Q-values.
      self._replay_next_target_net_outputs: The replayed next states' target
        Q-values (see Mnih et al., 2015 for details).
    """

    # _network_template instantiates the model and returns the network object.
    # The network object can be used to generate different outputs in the graph.
    # At each call to the network, the parameters will be reused.
    self.online_convnet = self._create_network(name='Online')
    self.target_convnet = self._create_network(name='Target')
    self._net_outputs = self.online_convnet(self.state_ph)
    # TODO(bellemare): Ties should be broken. They are unlikely to happen when
    # using a deep network, but may affect performance with a linear
    # approximation scheme.
    self._q_argmax = tf.argmax(self._net_outputs.q_values, axis=1)[0]
    self._replay_net_outputs = self.online_convnet(
        self._replay.transition['state'])
    self._replay_next_target_net_outputs = self.target_convnet(
        self._replay.transition['next_state'])

    self._q_argmax_sr = tf.argmax(self._net_outputs.q_values, axis=1)
    self.sr_convnet = SRNetwork(
        self.num_actions, atari_lib.NATURE_DQN_STACK_SIZE)
    # sr for states sampled
    self._sr_net_outputs = self.sr_convnet(self._replay.transition['state'])
    # sr for next_states sampled
    self._sr_net_outputs_next = self.sr_convnet(
        self._replay.transition['next_state'])
    # sr for current state and action
    self._sr_net_curr_state = self.sr_convnet(self.state_ph)

  def _build_replay_buffer(self, use_staging):
    return prioritized_replay_buffer.WrappedPrioritizedReplayBuffer(
        observation_shape=self.observation_shape,
        stack_size=self.stack_size,
        use_staging=use_staging,
        update_horizon=self.update_horizon,
        gamma=self.gamma,
        observation_dtype=self.observation_dtype.as_numpy_dtype)

  def _build_sr_train_op(self):
    feature = self._sr_net_outputs.feature
    decoded_state = self._sr_net_outputs.decoded_state

    loss_ae = tf.compat.v1.losses.huber_loss(
        self._replay.states, decoded_state, reduction=tf.losses.Reduction.NONE
    )
    srs = self._sr_net_outputs.sr_values
    indices = tf.transpose(
        tf.stack([self._replay.actions, tf.constant([i for i in range(32)])]))
    srs = tf.gather_nd(srs, indices)

    srs_next = self._sr_net_outputs_next.sr_values
    indices_next = tf.transpose(
        tf.stack([self._replay.next_actions, tf.constant([i for i in range(32)])]))
    srs_next = tf.gather_nd(srs_next, indices_next)

    assert feature.shape == srs_next.shape
    assert srs.shape == feature.shape

    loss_sr = tf.compat.v1.losses.mean_squared_error(
        srs, feature + self.gamma * srs_next
    )
    loss = loss_ae + loss_sr
    return self.optimizer.minimize(tf.reduce_mean(loss))

  def _build_train_op(self):
    """Builds a training op.
    Returns:
      train_op: An op performing one step of training from replay data.
    """
    replay_action_one_hot = tf.one_hot(
        self._replay.actions, self.num_actions, 1., 0., name='action_one_hot')
    replay_chosen_q = tf.reduce_sum(
        self._replay_net_outputs.q_values * replay_action_one_hot,
        axis=1,
        name='replay_chosen_q')

    # output from the SR network
    # note that the back prop of the q-loss should not take into account
    # the graph of the need term.
    curr_action = tf.stop_gradient(self._q_argmax_sr)
    sample_features = self._sr_net_outputs.feature
    curr_sr = self._sr_net_curr_state.sr_values
    curr_sr = tf.gather_nd(curr_sr, curr_action)
    need = tf.stop_gradient(
        tf.tensordot(curr_sr, sample_features, axes=[[1], [1]])[0]
    )

    target = tf.stop_gradient(self._build_target_q_op())
    loss = tf.compat.v1.losses.huber_loss(
        target, replay_chosen_q, reduction=tf.losses.Reduction.NONE)
    # The original prioritized experience replay uses a linear exponent
    # schedule 0.4 -> 1.0. Comparing the schedule to a fixed exponent of 0.5
    # on 5 games (Asterix, Pong, Q*Bert, Seaquest, Space Invaders) suggested
    # a fixed exponent actually performs better, except on Pong.
    probs = self._replay.transition['sampling_probabilities']
    loss_weights = 1.0 / tf.sqrt(probs + 1e-10)
    loss_weights /= tf.reduce_max(loss_weights)

    # Rainbow and prioritized replay are parametrized by an exponent alpha,
    # but in both cases it is set to 0.5 - for simplicity's sake we leave it
    # as is here, using the more direct tf.sqrt(). Taking the square root
    # "makes sense", as we are dealing with a squared loss.
    # Add a small nonzero value to the loss to avoid 0 priority items. While
    # technically this may be okay, setting all items to 0 priority will cause
    # troubles, and also result in 1.0 / 0.0 = NaN correction terms.
    update_priorities_op = self._replay.tf_set_priority(
        self._replay.indices, tf.sqrt(loss + 1e-10))

    # Weight the loss by the inverse priorities.
#     loss = loss_weights * loss * need
    loss = loss_weights * loss

    assert need.shape == loss.shape

    # TODO:
    # scheme 1:
    # - normalize feature
    # - if need smaller than 0, move all need values up
    need = need / tf.math.pow(tf.norm(sample_features, axis=1), 2)
    need = need - tf.minimum(tf.math.reduce_min(need), 0.0)

    loss_need = need * loss

    with tf.control_dependencies([update_priorities_op]):
      if self.summary_writer is not None:
        with tf.compat.v1.variable_scope('Losses'):
          tf.compat.v1.summary.scalar('HuberLoss', tf.reduce_mean(loss_need))
      return self.optimizer.minimize(tf.reduce_mean(loss_need))

  def _store_transition(self,
                        last_observation,
                        action,
                        reward,
                        is_terminal,
                        priority=None):
    priority = self._replay.memory.sum_tree.max_recorded_priority
    if not self.eval_mode:
      self._replay.add(last_observation, action, reward, is_terminal, priority)

  def _record_observation(self, observation):
    """Records an observation and update state.

    Extracts a frame from the observation vector and overwrites the oldest
    frame in the state buffer.

    Args:
      observation: numpy array, an observation from the environment.
    """
    # Set current observation. We do the reshaping to handle environments
    # without frame stacking.
    self._observation = np.reshape(observation, self.observation_shape)
    # Swap out the oldest frame with the current frame.
    self.state = np.roll(self.state, -1, axis=-1)
    self.state[0, ..., -1] = self._observation

  def _train_step(self):
    """Runs a single training step.

    Runs a training op if both:
      (1) A minimum number of frames have been added to the replay buffer.
      (2) `training_steps` is a multiple of `update_period`.

    Also, syncs weights from online to target network if training steps is a
    multiple of target update period.
    """
    # Run a train op at the rate of self.update_period if enough training steps
    # have been run. This matches the Nature DQN behaviour.
    if self._replay.memory.add_count > self.min_replay_history:
      if self.training_steps % self.update_period == 0:
        self._sess.run(self._train_op, {self.state_ph: self.state})

#         print(self._sess.run(self._sr_net_curr_state.feature, {self.state_ph: self.state}))
        self._sess.run(self._sr_train_op)
        if (self.summary_writer is not None and
            self.training_steps > 0 and
                self.training_steps % self.summary_writing_frequency == 0):
          summary = self._sess.run(self._merged_summaries)
          self.summary_writer.add_summary(summary, self.training_steps)

      if self.training_steps % self.target_update_period == 0:
        self._sess.run(self._sync_qt_ops)

    self.training_steps += 1


def create_prioritized_srdqn_agent(sess, environment, summary_writer=None):
  """The Runner class will expect a function of this type to create an agent."""
  return PrioritizedSRDQNAgent(sess, num_actions=environment.action_space.n)


prioritized_srdqn_config = """
import dopamine.discrete_domains.atari_lib
import dopamine.discrete_domains.run_experiment
import dopamine.agents.dqn.dqn_agent
import dopamine.replay_memory.prioritized_replay_buffer
import gin.tf.external_configurables

DQNAgent.gamma = 0.99
DQNAgent.update_horizon = 1
DQNAgent.min_replay_history = 20000  # agent steps
DQNAgent.update_period = 4
DQNAgent.target_update_period = 8000  # agent steps
DQNAgent.epsilon_train = 0.01
DQNAgent.epsilon_eval = 0.001
DQNAgent.epsilon_decay_period = 250000  # agent steps
DQNAgent.tf_device = '/gpu:0'  # use '/cpu:*' for non-GPU version
DQNAgent.optimizer = @tf.train.RMSPropOptimizer()

tf.train.RMSPropOptimizer.learning_rate = 0.00025
tf.train.RMSPropOptimizer.decay = 0.95
tf.train.RMSPropOptimizer.momentum = 0.0
tf.train.RMSPropOptimizer.epsilon = 0.00001
tf.train.RMSPropOptimizer.centered = True

atari_lib.create_atari_environment.game_name = '{}'
# Sticky actions with probability 0.25, as suggested by (Machado et al., 2017).
atari_lib.create_atari_environment.sticky_actions = True
create_agent.agent_name = 'dqn'
Runner.num_iterations = 200
Runner.training_steps = 250000  # agent steps
Runner.evaluation_steps = 125000  # agent steps
Runner.max_steps_per_episode = 27000  # agent steps

WrappedPrioritizedReplayBuffer.replay_capacity = 1000000
WrappedPrioritizedReplayBuffer.batch_size = 32
""".format(GAME)
gin.parse_config(prioritized_srdqn_config, skip_unknown=False)

# Create the runner class with this agent. We use very small numbers of steps
# to terminate quickly, as this is mostly meant for demonstrating how one can
# use the framework.
prioritized_srdqn_runner = run_experiment.TrainRunner(
    LOG_PATH, create_prioritized_srdqn_agent)

# @title Train MyRandomDQNAgent.
print('Will train agent, please be patient, may be a while...')
prioritized_srdqn_runner.run_experiment()
print('Done training!')
