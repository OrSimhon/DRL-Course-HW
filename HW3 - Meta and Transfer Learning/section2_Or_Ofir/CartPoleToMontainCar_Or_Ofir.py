######################### Or Simhon 315600486, Ofir Ben Moshe 315923151 #########################
# Actor Critic: MountainCar
import numpy as np
import gym
import tensorflow.compat.v1 as tf
from tensorboardX import SummaryWriter
import time
from sklearn.preprocessing import StandardScaler
import tensorflow_probability as tfp

tf.disable_v2_behavior()


class ActorNet:
    def __init__(self, state_size, action_size, CartPole_model_W1, CartPole_model_b1, name='ActorNetwork'):
        self.state_size = state_size
        self.action_size = action_size

        with tf.variable_scope(name):
            self.learning_rate = tf.placeholder(dtype=tf.float32, name='learning_rate')
            self.state = tf.placeholder(tf.float32, [None, self.state_size], name='state')
            self.action = tf.placeholder(tf.float32, name='action')
            self.delta = tf.placeholder(tf.float32, name='delta')
            self.I = tf.placeholder(tf.float32, name='I')

            self.W1 = CartPole_model_W1
            self.b1 = CartPole_model_b1
            self.W2 = tf.get_variable("W2", [64, 64],
                                      initializer=tf.keras.initializers.glorot_normal(seed=0))
            self.b2 = tf.get_variable("b2", [64], initializer=tf.zeros_initializer())
            self.W3 = tf.get_variable("W3", [64, self.action_size],
                                      initializer=tf.keras.initializers.glorot_normal(seed=0))
            self.b3 = tf.get_variable("b3", [self.action_size], initializer=tf.zeros_initializer())

            self.W_mean = tf.get_variable("W_mean", [self.action_size, 1],
                                          initializer=tf.keras.initializers.glorot_normal(seed=0))
            self.b_mean = tf.get_variable("b_mean", [1], initializer=tf.zeros_initializer())
            self.W_std = tf.get_variable("W_std", [self.action_size, 1],
                                         initializer=tf.keras.initializers.glorot_normal(seed=0))
            self.b_std = tf.get_variable("b_std", [1], initializer=tf.zeros_initializer())

            self.Z1 = tf.add(tf.matmul(self.state, self.W1), self.b1)
            self.A1 = tf.nn.relu(self.Z1)
            self.Z2 = tf.add(tf.matmul(self.A1, self.W2), self.b2)
            self.A2 = tf.nn.elu(self.Z2)
            self.output = tf.add(tf.matmul(self.A2, self.W3), self.b3)

            self.mean = tf.nn.tanh(tf.add(tf.matmul(self.output, self.W_mean), self.b_mean))
            self.std = tf.keras.activations.softplus(tf.add(tf.matmul(self.output, self.W_std), self.b_std)) + 1e-5
            self.norm_dist = tf.distributions.Normal(self.mean, self.std)
            self.actions_distribution = tf.clip_by_value(tf.squeeze(self.norm_dist.sample(1), axis=0), -1, 1)

            # actor_loss = -ln[Pr(A=a|S=s)] * I * delta
            self.loss = -tf.log(self.norm_dist.prob(self.action) + 1e-5) * self.delta * self.I
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def sample_action(self, sess, observation):
        actions_distribution = sess.run([self.actions_distribution], {self.state: observation})
        action = actions_distribution[0]
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
    def __init__(self, env, sess, hp, writer, scaler, saved_model_path):
        self.env = env
        self.general_observation_dim = 6
        self.general_action_dim = 3
        self.env_observation_dim = self.env.observation_space.shape[0]
        self.writer = writer
        self.gamma = hp['gamma']
        self.actor_lr = hp['actor_lr']
        self.critic_lr = hp['critic_lr']
        self.max_episodes = hp['max_episodes']
        self.sess = sess
        self.scaler = scaler

        W1, b1 = self.load_CartPole_model(saved_model_path)
        self.actor_net = ActorNet(state_size=self.general_observation_dim, action_size=self.general_action_dim,
                                  CartPole_model_W1=W1,
                                  CartPole_model_b1=b1)
        self.critic_net = CriticNet(state_size=self.general_observation_dim)

    def load_CartPole_model(self, saved_model_path):
        saver = tf.train.import_meta_graph(saved_model_path + "\CartPole\CartPoleModel.meta")
        saver.restore(self.sess, tf.train.latest_checkpoint(saved_model_path + "\CartPole"))
        graph = tf.get_default_graph()

        CartPole_model_W1 = tf.constant(graph.get_tensor_by_name('ActorNetwork/W1:0').eval())
        CartPole_model_b1 = tf.constant(graph.get_tensor_by_name('ActorNetwork/b1:0').eval())
        return CartPole_model_W1, CartPole_model_b1

    def update_networks(self, state, action, next_state, reward, done, I):

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
                     self.actor_net.action: np.squeeze(action), self.actor_net.learning_rate: self.actor_lr}
        _, actor_loss = self.sess.run([self.actor_net.optimizer, self.actor_net.loss], feed_dict)

        return actor_loss

    def test_agent(self, visualize='True'):
        s = self.env.reset()
        s = np.pad(self.scaler.transform(s.reshape(1, -1)).flatten(),
                   (0, self.general_observation_dim - self.env_observation_dim), 'constant')
        total_reward = 0.0
        finish = False

        while not finish:
            if visualize:
                self.env.render()
            a = self.actor_net.sample_action(self.sess, np.atleast_2d(s))
            s, r, finish, _ = self.env.step(a)
            s = np.pad(self.scaler.transform(s.reshape(1, -1)).flatten(),
                       (0, self.general_observation_dim - self.env_observation_dim), 'constant')
            total_reward += r
        print("Total reward in the test: %.2f" % total_reward)
        self.env.close()

    def train(self):
        start_time = time.time()
        self.sess.run(tf.global_variables_initializer())
        max_score = 90
        episode_loss = []
        average_score = []
        total_steps = 0
        initiail_actor_lr = self.actor_lr
        initiail_critic_lr = self.critic_lr
        for episode in range(self.max_episodes):
            state = self.env.reset()
            state = np.pad(self.scaler.transform(state.reshape(1, -1)).flatten(),
                           (0, self.general_observation_dim - self.env_observation_dim), 'constant')

            done = False
            episode_score = 0
            I = 1  # Discount Factor

            # Learning rates decay
            if episode % 60 == 0 and episode > 0:
                self.actor_lr *= 0.7
                self.critic_lr *= 0.7

            while not done:
                total_steps += 1
                action = self.actor_net.sample_action(self.sess, np.atleast_2d(state))
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.pad(self.scaler.transform(next_state.reshape(1, -1)).flatten(),
                                    (0, self.general_observation_dim - self.env_observation_dim), 'constant')
                action_vector = np.zeros(self.general_action_dim)
                action_vector[0] = action
                episode_score += reward

                actor_loss = self.update_networks(np.atleast_2d(state), action,
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

            if max(average_score[-5:]) < 0 and episode > 15:
                I = 1
                self.actor_lr = initiail_actor_lr
                self.critic_lr = initiail_critic_lr
                self.sess.run(tf.global_variables_initializer())
            elif np.mean(average_score[-100:]) >= max_score and episode > 99:
                print(
                    "\nGreat!! "
                    "You win after: {} Episodes\n"
                    "Average reward in the last 100 episodes: {:.2f}".format(episode + 1,
                                                                             np.mean(average_score[-100:])))
                saver = tf.train.Saver()
                saver.save(self.sess, 'SavedModels/MountainCarModel')
                break

        training_time = (time.time() - start_time) / 60
        print("Training complete after {:.2} minutes".format(training_time))
        self.writer.close()
        self.test_agent()


if __name__ == '__main__':
    env = gym.make("MountainCarContinuous-v0")

    best_hyper_parameters = {
        'critic_lr': 0.001,
        'actor_lr': 0.00002,
        'gamma': 0.99,
        'max_episodes': 1300
    }

    writer = SummaryWriter()
    tf.reset_default_graph()
    sample_states = np.array([env.observation_space.sample() for x in range(10_000)])
    scaler = StandardScaler()
    scaler.fit(sample_states)
    saved_model_path = 'G:\My Drive\Python Projects\TensorFlowProjects\DRL Course - Gilad\Or and Ofir\HW3\section1_Or_Ofir\SavedModels'
    with tf.Session() as sess:
        model = ActorCritic_Agent(env=env, sess=sess, hp=best_hyper_parameters, writer=writer, scaler=scaler,
                                  saved_model_path=saved_model_path)
        model.train()
