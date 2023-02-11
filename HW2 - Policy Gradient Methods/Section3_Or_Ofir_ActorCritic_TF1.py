import gym
import numpy as np
import tensorflow.compat.v1 as tf
from tensorboardX import SummaryWriter

tf.disable_v2_behavior()
env = gym.make('CartPole-v1')

# Define hyperparameters
state_size = 4
action_size = env.action_space.n

max_episodes = 5000
max_steps = 501
discount_factor = 0.99
policy_learning_rate = 0.001
value_learning_rate = 0.01

render = False


class ValueNetwork:
    """
    Valeu Network class trying to estimate the value function (used as the critic)
    """

    def __init__(self, state_size, name='value_network'):
        self.state_size = state_size

        with tf.variable_scope(name):
            self.learning_rate = tf.placeholder(dtype=tf.float32, name="learning_rate")
            self.state = tf.placeholder(tf.float32, [None, self.state_size], name='state')
            self.target = tf.placeholder(tf.float32, name="total_rewards")

            self.W1 = tf.get_variable("W1", [self.state_size, 64],
                                      initializer=tf.keras.initializers.glorot_normal(seed=0))
            self.b1 = tf.get_variable("b1", [64], initializer=tf.zeros_initializer())
            self.W2 = tf.get_variable("W2", [64, 1], initializer=tf.keras.initializers.glorot_normal(seed=0))
            self.b2 = tf.get_variable("b2", [1], initializer=tf.zeros_initializer())

            self.Z1 = tf.add(tf.matmul(self.state, self.W1), self.b1)
            self.A1 = tf.nn.relu(self.Z1)
            self.output = tf.add(tf.matmul(self.A1, self.W2), self.b2)

            self.value = tf.squeeze(self.output)

            # Trying to estimate the value value
            self.loss = tf.reduce_mean(tf.square(self.target - self.value))

            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)


class PolicyNetwork:
    def __init__(self, state_size, action_size, name='policy_network'):
        self.state_size = state_size
        self.action_size = action_size

        with tf.variable_scope(name):
            self.learning_rate = tf.placeholder(dtype=tf.float32, name="learning_rate")
            self.state = tf.placeholder(tf.float32, [None, self.state_size], name="state")
            self.action = tf.placeholder(tf.int32, [self.action_size], name="action")
            self.delta_I = tf.placeholder(tf.float32, name="delta_I")

            self.W1 = tf.get_variable("W1", [self.state_size, 64],
                                      initializer=tf.keras.initializers.glorot_normal(seed=0))
            self.b1 = tf.get_variable("b1", [64], initializer=tf.zeros_initializer())
            self.W2 = tf.get_variable("W2", [64, self.action_size],
                                      initializer=tf.keras.initializers.glorot_normal(seed=0))
            self.b2 = tf.get_variable("b2", [self.action_size], initializer=tf.zeros_initializer())

            self.Z1 = tf.add(tf.matmul(self.state, self.W1), self.b1)
            self.A1 = tf.nn.relu(self.Z1)
            self.output = tf.add(tf.matmul(self.A1, self.W2), self.b2)

            # Softmax probability distribution over actions
            self.actions_distribution = tf.squeeze(tf.nn.softmax(self.output))
            # Loss with negative log probability
            self.neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output, labels=self.action)
            self.loss = tf.reduce_mean(self.neg_log_prob * self.delta_I)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)


class ActorCritic:
    """
    A class to manage the actor-critic updates
    """

    def __init__(self, state_size, action_size, discount_factor, sess,
                 tb_writer):
        """
        :param state_size: The size of each state.
        :param action_size: Amount of possible actions.
        :param discount_factor: Discount factor for future rewards.
        :param sess: A tensorflow session
        :param tb_writer: the writer for tensorboardX logger
        """
        self.state_size = state_size
        self.action_size = action_size
        self.actor_learning_rate = policy_learning_rate
        self.critic_learning_rate = value_learning_rate
        self.discount_factor = discount_factor
        self.sess = sess
        self.tb_writer = tb_writer

        # Init the actor
        self.actor = PolicyNetwork(self.state_size, self.action_size)

        # Init the critic
        self.critic = ValueNetwork(self.state_size)

    def train(self, state, action_one_hot, reward, next_state, done, all_steps_counter, policy_learning_rate,
              value_learning_rate, I):
        """
        A function to update the actor and critic networks

        :param state: value of current state.
        :param action_one_hot: The chosen action.
        :param reward: The reward of the step.
        :param next_state: The next state from the environment.
        :param done: Is the step is final state?
        :param all_steps_counter: counter of all steps until now.
        :param policy_learning_rate: Learning rate for the actor (used for the lr decay)
        :param value_learning_rate: Learning rate for the critic (used for the lr decay)
        :param I: value of I.
        :return:
        """

        # Predict the value of the current state and next state
        value = self.sess.run(self.critic.value, {self.critic.state: state})
        value_next = self.sess.run(self.critic.value, {self.critic.state: next_state})

        # Calculate the target values to train the networks, depends if the state is final or not
        if done:
            delta = reward - value
            value_target = reward
        else:
            delta = reward + self.discount_factor * value_next - value
            value_target = reward + self.discount_factor * value_next

        # Update the critic
        feed_dict_val = {self.critic.state: state, self.critic.target: value_target,
                         self.critic.learning_rate: value_learning_rate}
        _, val_loss = sess.run([self.critic.optimizer, self.critic.loss], feed_dict_val)
        self.tb_writer.add_scalar('value_network_loss', val_loss, all_steps_counter)

        # Update the actor
        feed_dict = {self.actor.state: state, self.actor.delta_I: I * delta, self.actor.action: action_one_hot,
                     self.actor.learning_rate: policy_learning_rate}
        _, loss = self.sess.run([self.actor.optimizer, self.actor.loss], feed_dict)
        self.tb_writer.add_scalar('policy_network_loss', loss, all_steps_counter)


tf.reset_default_graph()

writer = SummaryWriter(comment="policy_grad_actor")  # NEW

# Start training the agent with REINFORCE algorithm
with tf.Session() as sess:
    actor_critic_model = ActorCritic(state_size, action_size, discount_factor, sess,
                                     writer)
    sess.run(tf.global_variables_initializer())
    solved = False
    episode_rewards = np.zeros(max_episodes)
    average_rewards = 0.0
    all_steps_counter = 0

    for episode in range(max_episodes):
        I = 1

        # Learning rates decay
        if episode % 60 == 0 and episode > 0:
            policy_learning_rate *= 0.7
            value_learning_rate *= 0.7

        state = env.reset()
        state = state.reshape([1, state_size])
        episode_transitions = []

        for step in range(max_steps):
            actions_distribution = sess.run(actor_critic_model.actor.actions_distribution,
                                            {actor_critic_model.actor.state: state})
            action = np.random.choice(np.arange(len(actions_distribution)), p=actions_distribution)
            next_state, reward, done, _ = env.step(action)
            next_state = next_state.reshape([1, state_size])

            if render:
                env.render()

            action_one_hot = np.zeros(action_size)
            action_one_hot[action] = 1

            episode_rewards[episode] += reward

            # Train both networks based on the transition
            actor_critic_model.train(state, action_one_hot, reward, next_state, done, all_steps_counter,
                                     policy_learning_rate, value_learning_rate, I)

            if done:
                if episode > 98:
                    # Check if solved
                    average_rewards = np.mean(episode_rewards[(episode - 99):episode + 1])
                print("Episode {} Reward: {} Average over 100 episodes: {}".format(episode, episode_rewards[episode],
                                                                                   round(average_rewards, 2)))
                writer.add_scalar('reward', episode_rewards[episode], episode)  # NEW
                writer.add_scalar('avg_reward_per_100_steps', round(average_rewards, 2), episode)  # NEW
                if average_rewards > 475:
                    print(' Solved at episode: ' + str(episode))
                    solved = True
                break

            state = next_state
            all_steps_counter += 1
            I *= discount_factor

        if solved:
            break
