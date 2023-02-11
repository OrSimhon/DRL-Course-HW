######################### Or Simhon 315600486, Ofir Ben Moshe 315923151 #########################
# Policy Gradient (REINFORCE) with baseline: Cart Pole
import numpy as np
import gym
import tensorflow as tf
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense
from tensorboardX import SummaryWriter
import tensorflow_probability as tfp


class PolicyNet:
    def __init__(self, input_size, output_size):
        """
        Initialize Policy Network: pi(S=s) = prob(A|S=s)
        :param input_size: Input size to the Network
        :param output_size: Output size to the Network
        """
        self.model = Sequential(
            layers=[
                Input(shape=(input_size,)),
                Dense(units=64, activation='relu', name='relu_layer'),
                Dense(units=output_size, activation='softmax', name='softmax_layer')
            ],
            name='Policy')

    def actions_distribution(self, observations):
        """
        Get the actions probabilities given states
        :param observations: States representation
        :return: Actions probabilities
        """
        probs = self.model(observations)  # Send to PolicyNet states and get back probabilities tensor with 2 columns
        action_probs = tfp.distributions.Categorical(probs=probs)  # Change the type of probs (same values)
        return action_probs

    def sample_action(self, observation):
        """
        Sample action from the policy NN
        :param observation: State representation
        :return: actions
        """
        return self.actions_distribution(observation).sample().numpy()[0]


class BaselineNet:
    def __init__(self, input_size, output_size):
        """
        Initialize Baseline Network (describe the Value function): V(S=s)
        :param input_size: Input size to the Network
        :param output_size: Output size to the Network
        """
        self.model = Sequential(
            layers=[
                Input(shape=(input_size,)),
                Dense(units=64, activation='relu', name='relu_layer'),
                Dense(units=output_size, activation='linear', name='linear_layer')
            ], name='baseline')

    def forward(self, observations):
        """
        Send states to the Baseline NN and get back V(S=s)
        :param observations: States
        :return: V(S=s)
        """
        output = tf.squeeze(self.model(observations))
        return output


class REINFORCE_with_Baseline:
    def __init__(self, env, hp, writer):
        self.env = env
        self.observation_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        self.writer = writer

        self.gamma = hp['gamma']
        self.policy_lr = hp['policy_lr']
        self.baseline_lr = hp['baseline_lr']
        self.max_episodes = hp['max_episodes']

        #  2 Networks, 2 Optimizers (different learning rates)
        self.policy_net = PolicyNet(input_size=self.observation_dim, output_size=self.action_dim)
        self.baseline_net = BaselineNet(input_size=self.observation_dim, output_size=1)
        self.policy_optimizer = tf.keras.optimizers.Adam(learning_rate=self.policy_lr)
        self.baseline_optimizer = tf.keras.optimizers.Adam(learning_rate=self.baseline_lr)

    def generate_episode(self):
        """
        Generate an episode
        Returns: trajectory as different arrays (rewards as list)
        """
        state = self.env.reset()
        states = []
        actions = []
        rewards = []
        done = False
        cumulative_reward = 0
        while not done:
            action = self.policy_net.sample_action(np.atleast_2d(state))
            next_state, reward, done, _ = self.env.step(action)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            state = next_state
            cumulative_reward += reward

        return np.array(states), np.array(actions), rewards

    def get_returns(self, rewards):
        """
        Calculate Gt for each step in the last trajectory
        :param rewards: list of all the received rewards in the last trajectory
        :return: array of Gt for each step in the last episode
        """
        returns = []
        reversed_rewards = np.flip(rewards, 0)
        g_t = 0
        for r in reversed_rewards:
            g_t = r + self.gamma * g_t
            returns.insert(0, g_t)
        return np.array(returns)

    def get_deltas(self, Gts, states):
        """
        Get delta, namely Gt-v(s) for each step in the trajectory
        :param Gts:
        :param states:
        :return:
        """
        values = self.baseline_net.forward(states).numpy()
        return Gts - values

    def update_networks(self, states, actions, targets):
        """
        Update the policy network.
        :param states:
        :param actions:
        :param targets:
        :return:
        """
        deltas = self.get_deltas(Gts=targets, states=states)
        states = tf.convert_to_tensor(states)
        actions = tf.convert_to_tensor(actions)
        deltas = tf.convert_to_tensor(deltas)

        with tf.GradientTape(persistent=True) as tape:
            v = self.baseline_net.forward(observations=states)  # V(S=states) from Baseline NN
            action_probs = self.policy_net.actions_distribution(states)  # Actions probabilities: Pr(A|S=states)
            log_prob = action_probs.log_prob(actions)  # log of the taken action a: ln[Pr(A=a|S=states)]

            # policy_loss = -ln[Pr(A=a|S=states)] * delta
            policy_loss = -tf.math.reduce_mean(log_prob * tf.cast(deltas, tf.float32))

            # baseline_loss = (Gt-v) ** 2
            baseline_loss = tf.keras.losses.mean_squared_error(y_true=targets, y_pred=v)

        policy_grads = tape.gradient(policy_loss, self.policy_net.model.trainable_weights)
        self.policy_optimizer.apply_gradients(zip(policy_grads, self.policy_net.model.trainable_weights))

        critic_grads = tape.gradient(baseline_loss, self.baseline_net.model.trainable_weights)
        self.baseline_optimizer.apply_gradients(zip(critic_grads, self.baseline_net.model.trainable_weights))

        return policy_loss.numpy()

    def test_agent(self, visualize='True'):
        s = self.env.reset()
        total_reward = 0.0
        finish = False

        while not finish:
            if visualize:
                self.env.render()
            a = self.policy_net.sample_action(np.atleast_2d(s))
            s, r, finish, _ = self.env.step(a)
            total_reward += r
        print("Total reward in the test: %.2f" % total_reward)
        env.close()

    def train(self):
        """
        Train the agent - main loop
        """
        max_score = 475.0  # Average score to reach in 100 consecutive episodes
        average_score = []
        for episode in range(self.max_episodes):
            states, actions, rewards = self.generate_episode()
            returns = self.get_returns(rewards)
            policy_loss = self.update_networks(states, actions, targets=returns)

            episode_score = np.sum(rewards)
            average_score.append(episode_score)

            print(
                "Episode {} | Reward: {:04.2f} | Average over 100 episodes: {:04.2f}".format(episode + 1, episode_score,
                                                                                             np.mean(
                                                                                                 average_score[-100:])))

            self.writer.add_scalar('Total reward per episode', episode_score, episode)
            self.writer.add_scalar('Policy Loss per episode', policy_loss, episode)
            self.writer.add_scalar('Mean reward in the last 100 episodes', np.mean(average_score[-100:]), episode)

            if np.mean(average_score[-100:]) >= max_score:
                print(
                    "\nGreat!! "
                    "You win after: {} Episodes\n"
                    "Average reward in the last 100 episodes: {}".format(episode + 1, np.mean(average_score[-100:])))
                break
        self.writer.close()
        print("Training complete")
        self.test_agent()


if __name__ == '__main__':
    env = gym.make("CartPole-v1")

    # This HP win after 537 Episodes
    best_hyper_parameters = {
        'policy_lr': 0.002,
        'baseline_lr': 0.002,
        'gamma': 0.99,
        'max_episodes': 1500
    }
    writer = SummaryWriter()
    model = REINFORCE_with_Baseline(env=env, hp=best_hyper_parameters, writer=writer)
    model.train()
