######################### Or Simhon 315600486, Ofir Ben Moshe 315923151 #########################
# Actor Critic: Acrobot
import numpy as np
import gym
import tensorflow.compat.v1 as tf
from tensorboardX import SummaryWriter
import time

tf.disable_v2_behavior()


class ActorNet:
    def __init__(self, state_size, action_size, name='ActorNetwork'):
        self.state_size = state_size
        self.action_size = action_size

        with tf.variable_scope(name):
            self.learning_rate = tf.placeholder(dtype=tf.float32, name='learning_rate')
            self.state = tf.placeholder(tf.float32, [None, self.state_size], name='state')
            self.action = tf.placeholder(tf.int32, [self.action_size], name='action')
            self.delta = tf.placeholder(tf.float32, name='delta')
            self.I = tf.placeholder(tf.float32, name='I')

            self.W1 = tf.get_variable('W1', [self.state_size, 64],
                                      initializer=tf.keras.initializers.glorot_normal(seed=0))
            self.b1 = tf.get_variable("b1", [64], initializer=tf.zeros_initializer())
            self.W2 = tf.get_variable("W2", [64, self.action_size],
                                      initializer=tf.keras.initializers.glorot_normal(seed=0))
            self.b2 = tf.get_variable("b2", [self.action_size], initializer=tf.zeros_initializer())

            self.Z1 = tf.add(tf.matmul(self.state, self.W1), self.b1)
            self.A1 = tf.nn.relu(self.Z1)
            self.output = tf.add(tf.matmul(self.A1, self.W2), self.b2)

            # Actions probabilities: Pr(A|S=s)
            self.actions_distribution = tf.squeeze(tf.nn.softmax(self.output))
            # log of the taken action a: ln[Pr(A=a|S=s)]
            self.neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output, labels=self.action)
            # actor_loss = -ln[Pr(A=a|S=s)] * I * delta
            self.loss = tf.reduce_mean(self.neg_log_prob * self.delta * self.I)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def sample_action(self, sess, observation, env_action_dim):
        actions_distribution = sess.run(self.actions_distribution,
                                        {self.state: observation})[:env_action_dim]

        actions_distribution = np.ones(env_action_dim) / env_action_dim if sum(actions_distribution == 0) else [
            x * (1 / (float(sum(actions_distribution)))) for x in actions_distribution]
        action = np.random.choice(np.arange(env_action_dim), p=actions_distribution)

        return action


class CriticNet:
    def __init__(self, state_size, name='CriticNetwork'):
        self.state_size = state_size

        with tf.variable_scope(name):
            self.learning_rate = tf.placeholder(dtype=tf.float32, name="learning_rate")
            self.state = tf.placeholder(tf.float32, [None, self.state_size], name='state')
            self.target = tf.placeholder(tf.float32, name="Gt")
            self.I = tf.placeholder(tf.float32, name='I')

            self.W1 = tf.get_variable('W1', [self.state_size, 64],
                                      initializer=tf.keras.initializers.glorot_normal(seed=0))
            self.b1 = tf.get_variable("b1", [64], initializer=tf.zeros_initializer())
            self.W2 = tf.get_variable("W2", [64, 1],
                                      initializer=tf.keras.initializers.glorot_normal(seed=0))
            self.b2 = tf.get_variable("b2", [1], initializer=tf.zeros_initializer())

            self.Z1 = tf.add(tf.matmul(self.state, self.W1), self.b1)
            self.A1 = tf.nn.relu(self.Z1)
            self.value = tf.squeeze(tf.add(tf.matmul(self.A1, self.W2), self.b2))

            # critic_loss = ((target - v) * I) ** 2
            self.loss = tf.reduce_mean(tf.square((self.target - self.value) * self.I))
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)


class ActorCritic_Agent:
    def __init__(self, env, sess, hp, writer):
        self.env = env
        self.general_observation_dim = 6
        self.general_action_dim = 3
        self.env_observation_dim = self.env.observation_space.shape[0]
        self.env_action_dim = self.env.action_space.n
        self.writer = writer
        self.gamma = hp['gamma']
        self.actor_lr = hp['actor_lr']
        self.critic_lr = hp['critic_lr']
        self.max_episodes = hp['max_episodes']
        self.sess = sess

        self.actor_net = ActorNet(state_size=self.general_observation_dim, action_size=self.general_action_dim)
        self.critic_net = CriticNet(state_size=self.general_observation_dim)

    def update_networks(self, state, action_one_hot, next_state, reward, done, I):

        # Predict the value of the current state and next state
        v = self.sess.run(self.critic_net.value, {self.critic_net.state: state})  # V(s) from Critic NN
        v_next = self.sess.run(self.critic_net.value, {self.critic_net.state: next_state})  # V(s') from Critic NN

        target = np.atleast_1d(reward + self.gamma * v_next * (1 - int(done)))  # Gt = r + gamma * v(s')
        delta = target - v

        # Update the critic
        feed_dict_val = {self.critic_net.state: state, self.critic_net.target: target, self.critic_net.I: I,
                         self.critic_net.learning_rate: self.critic_lr}
        _, val_loss = self.sess.run([self.critic_net.optimizer, self.critic_net.loss], feed_dict_val)

        # Update the actor
        feed_dict = {self.actor_net.state: state, self.actor_net.delta: delta, self.actor_net.I: I,
                     self.actor_net.action: action_one_hot,
                     self.actor_net.learning_rate: self.actor_lr}
        _, actor_loss = self.sess.run([self.actor_net.optimizer, self.actor_net.loss], feed_dict)

        return actor_loss

    def test_agent(self, visualize='True'):

        s = np.pad(self.env.reset(), (0, self.general_observation_dim - self.env_observation_dim), 'constant')
        total_reward = 0.0
        finish = False

        while not finish:
            if visualize:
                self.env.render()
            a = self.actor_net.sample_action(self.sess, np.atleast_2d(s), self.env_action_dim)
            s, r, finish, _ = self.env.step(a)
            s = np.pad(s, (0, self.general_observation_dim - self.env_observation_dim), 'constant')
            total_reward += r
        print("Total reward in the test: %.2f" % total_reward)
        self.env.close()

    def train(self):
        start_time = time.time()
        self.sess.run(tf.global_variables_initializer())
        max_score = -100
        episode_loss = []
        average_score = []
        total_steps = 0

        for episode in range(self.max_episodes):
            state = np.pad(self.env.reset(), (0, self.general_observation_dim - self.env_observation_dim), 'constant')
            done = False
            episode_score = 0
            I = 1  # Discount Factor

            # Learning rates decay
            if episode % 60 == 0 and episode > 0:
                self.actor_lr *= 0.7
                self.critic_lr *= 0.7

            while not done:
                total_steps += 1
                action = self.actor_net.sample_action(self.sess, np.atleast_2d(state), self.env_action_dim)
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.pad(next_state, (0, self.general_observation_dim - self.env_observation_dim),
                                    'constant')
                action_one_hot = np.zeros(self.general_action_dim)
                action_one_hot[action] = 1
                episode_score += reward

                actor_loss = self.update_networks(np.atleast_2d(state), action_one_hot,
                                                  np.atleast_2d(next_state), reward, done, I)

                self.writer.add_scalar('Actor Loss per step', actor_loss, total_steps)
                episode_loss.append(actor_loss)
                state = next_state
                I *= self.gamma
            average_score.append(episode_score)
            print(
                "Episode {} | Reward: {:04.2f} | Average over 100 episodes: {:04.2f}".format(episode + 1,
                                                                                             episode_score,
                                                                                             np.mean(
                                                                                                 average_score[
                                                                                                 -100:])))
            self.writer.add_scalar('Total reward per episode', episode_score, episode)
            self.writer.add_scalar('Mean reward in the last 100 episodes', np.mean(average_score[-100:]), episode)

            if np.mean(average_score[-100:]) >= max_score and episode > 100:
                print(
                    "\nGreat!! "
                    "You win after: {} Episodes\n"
                    "Average reward in the last 100 episodes: {}".format(episode + 1,
                                                                         np.mean(average_score[-100:])))
                saver = tf.train.Saver()
                saver.save(self.sess, 'SavedModels/AcrobotModel')
                break

        training_time = (time.time() - start_time) / 60
        print("Training complete after {:.2} minutes".format(training_time))
        self.writer.close()
        self.test_agent()


if __name__ == '__main__':
    env = gym.make("Acrobot-v1")

    best_hyper_parameters = {
        'critic_lr': 0.01,
        'actor_lr': 0.001,
        'gamma': 0.99,
        'max_episodes': 1300
    }
    writer = SummaryWriter()
    tf.reset_default_graph()
    with tf.Session() as sess:
        model = ActorCritic_Agent(env=env, sess=sess, hp=best_hyper_parameters, writer=writer)
        model.train()
