# @title Necessary imports and globals.
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
LOG_PATH = os.path.join(BASE_PATH, 'prioritized_dqn', GAME)

# @title Create the DQN with prioritized replay
from dopamine.replay_memory import prioritized_replay_buffer
import tensorflow as tf

class PrioritizedDQNAgent(dqn_agent.DQNAgent):
  def __init__(self, sess, num_actions):
    """This maintains all the DQN default argument values."""
    super().__init__(sess, num_actions, tf_device='/gpu:0')
    self._replay_scheme = 'prioritized'

  def _build_replay_buffer(self, use_staging):
    return prioritized_replay_buffer.WrappedPrioritizedReplayBuffer(
        observation_shape=self.observation_shape,
        stack_size=self.stack_size,
        use_staging=use_staging,
        update_horizon=self.update_horizon,
        gamma=self.gamma,
        observation_dtype=self.observation_dtype.as_numpy_dtype)

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
    loss = loss_weights * loss
    
    with tf.control_dependencies([update_priorities_op]):
      if self.summary_writer is not None:
        with tf.compat.v1.variable_scope('Losses'):
          tf.compat.v1.summary.scalar('HuberLoss', tf.reduce_mean(loss))
      return self.optimizer.minimize(tf.reduce_mean(loss))

  def _store_transition(self,
                        last_observation,
                        action,
                        reward,
                        is_terminal,
                        priority=None):
    priority = self._replay.memory.sum_tree.max_recorded_priority
    if not self.eval_mode:
      self._replay.add(last_observation, action, reward, is_terminal, priority)
def create_prioritized_dqn_agent(sess, environment, summary_writer=None):
  """The Runner class will expect a function of this type to create an agent."""
  return PrioritizedDQNAgent(sess, num_actions=environment.action_space.n)

prioritized_dqn_config = """
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
gin.parse_config(prioritized_dqn_config, skip_unknown=False)

# Create the runner class with this agent. We use very small numbers of steps
# to terminate quickly, as this is mostly meant for demonstrating how one can
# use the framework.
prioritized_dqn_runner = run_experiment.TrainRunner(LOG_PATH, create_prioritized_dqn_agent)

# @title Train MyRandomDQNAgent.
print('Will train agent, please be patient, may be a while...')
prioritized_dqn_runner.run_experiment()
print('Done training!')