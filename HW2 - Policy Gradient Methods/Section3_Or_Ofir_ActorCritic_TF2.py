######################### Or Simhon 315600486, Ofir Ben Moshe 315923151 #########################
# Actor Critic: Cart Pole
import numpy as np
import gym
import tensorflow as tf
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense
from tensorboardX import SummaryWriter
import tensorflow_probability as tfp


class ActorNet:
    def __init__(self, input_size, output_size):
        """
        Initialize Actor Network (describe the policy): pi(S=s) = prob(A|S=s)
        :param input_size: Input size to the Network
        :param output_size: Output size to the Network
        """
        self.model = Sequential(
            layers=[
                Input(shape=(input_size,)),
                Dense(units=64, activation='relu', name='relu_layer'),
                Dense(units=output_size, activation='softmax', name='softmax_layer')
            ],
            name='Actor')

    def action_distribution(self, observation):
        """
        Get the actions probabilities given state
        :param observation: State representation
        :return: Actions probabilities
        """
        probs = self.model(observation)  # Send to ActorNet state and get back probability tensor with size 2
        action_probs = tfp.distributions.Categorical(probs=probs)  # Change the type of probs (same values)
        return action_probs

    def sample_action(self, observation):
        """
        Sample action from the Actor NN policy
        :param observation: State representation
        :return: action
        """
        return self.action_distribution(observation).sample().numpy()[0]  # Sample action from actions probs


class CriticNet:
    def __init__(self, input_size, output_size):
        """
        Initialize Critic Network (describe the Value function): V(S=s)
        :param input_size: Input size to the Network
        :param output_size: Output size to the Network
        """
        self.model = Sequential(
            layers=[
                Input(shape=(input_size,)),
                Dense(units=64, activation='relu', name='relu_layer'),
                Dense(units=output_size, activation='linear', name='linear_layer')
            ], name='Critic')

    def forward(self, observation):
        """
        Send state to the Critic NN and get back V(S=s)
        :param observation: State
        :return: V(S=s)
        """
        output = tf.squeeze(self.model(observation))
        return output


class ActorCritic_Agent:
    def __init__(self, env, hp, writer):
        self.env = env
        self.observation_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        self.writer = writer
        self.gamma = hp['gamma']
        self.actor_lr = hp['actor_lr']
        self.critic_lr = hp['critic_lr']
        self.max_episodes = hp['max_episodes']

        #  2 Networks, 2 Optimizers (different learning rates)
        self.actor_net = ActorNet(input_size=self.observation_dim, output_size=self.action_dim)
        self.critic_net = CriticNet(input_size=self.observation_dim, output_size=1)
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=self.actor_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=self.critic_lr)

    def update_networks(self, state, action, next_state, reward, done, I):
        """
        Update the networks
        :param state: Current state s
        :param action: The action a taken from state s
        :param next_state: The new state s'
        :param reward: The immediate reward
        :param done: Check for terminal state sT
        :param I: Discount factor - change NN's less along the learning
        :return: Actor loss
        """
        state = tf.convert_to_tensor(state)
        next_state = tf.convert_to_tensor(next_state)

        with tf.GradientTape(persistent=True) as tape:
            v = self.critic_net.forward(state)  # V(s) from Critic NN
            v_next = self.critic_net.forward(next_state)  # V(s') from Critic NN
            target = np.atleast_1d(reward + self.gamma * v_next * (1 - int(done)))  # Gt = r + gamma * v(s')
            delta = target - v

            action_probs = self.actor_net.action_distribution(state)  # Actions probabilities: Pr(A|S=s)
            log_prob = action_probs.log_prob(action)  # log of the taken action a: ln[Pr(A=a|S=s)]

            # actor_loss = -ln[Pr(A=a|S=s)] * I * delta
            actor_loss = -tf.math.reduce_mean(log_prob * tf.cast(np.atleast_1d(I * delta), tf.float32))

            # critic_loss = ((target - v) * I) ** 2
            critic_loss = tf.keras.losses.mean_squared_error(y_true=I * target, y_pred=I * v)

        actor_grads = tape.gradient(actor_loss, self.actor_net.model.trainable_weights)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor_net.model.trainable_weights))

        critic_grads = tape.gradient(critic_loss, self.critic_net.model.trainable_weights)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic_net.model.trainable_weights))

        return actor_loss.numpy()

    def test_agent(self, visualize='True'):
        s = self.env.reset()
        total_reward = 0.0
        finish = False

        while not finish:
            if visualize:
                self.env.render()
            a = self.actor_net.sample_action(np.atleast_2d(s))
            s, r, finish, _ = self.env.step(a)
            total_reward += r
        print("Total reward in the test: %.2f" % total_reward)
        env.close()

    def train(self):
        max_score = 475.0
        episode_loss = []
        average_score = []
        total_steps = 0
        for episode in range(self.max_episodes):
            state = self.env.reset()
            done = False
            episode_score = 0
            I = 1  # Discount Factor

            # Learning rates decay
            if episode % 60 == 0 and episode > 0:
                self.actor_lr *= 0.7
                self.critic_lr *= 0.7

            while not done:
                total_steps += 1
                action = self.actor_net.sample_action(np.atleast_2d(state))
                next_state, reward, done, _ = self.env.step(action)
                episode_score += reward
                actor_loss = self.update_networks(np.atleast_2d(state), action,
                                                  np.atleast_2d(next_state), reward, done, I)

                writer.add_scalar('Actor Loss per step', actor_loss, total_steps)
                episode_loss.append(actor_loss)
                state = next_state
                I *= self.gamma
            average_score.append(episode_score)
            print(
                "Episode {} | Reward: {:04.2f} | Average over 100 episodes: {:04.2f}".format(episode + 1, episode_score,
                                                                                             np.mean(
                                                                                                 average_score[-100:])))
            writer.add_scalar('Total reward per episode', episode_score, episode)
            writer.add_scalar('Mean reward in the last 100 episodes', np.mean(average_score[-100:]), episode)

            if np.mean(average_score[-100:]) >= max_score:
                print(
                    "\nGreat!! "
                    "You win after: {} Episodes\n"
                    "Average reward in the last 100 episodes: {}".format(episode + 1, np.mean(average_score[-100:])))
                break

        print("Training complete")
        writer.close()
        self.test_agent()


if __name__ == '__main__':
    env = gym.make("CartPole-v1")

    # This HP win after 431 Episodes
    best_hyper_parameters = {
        'critic_lr': 0.01,
        'actor_lr': 0.001,
        'gamma': 0.99,
        'max_episodes': 400
    }
    writer = SummaryWriter()
    model = ActorCritic_Agent(env=env, hp=best_hyper_parameters, writer=writer)
    model.train()
