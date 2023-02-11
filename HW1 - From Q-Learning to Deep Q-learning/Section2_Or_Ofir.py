######################### Or Simhon 315600486, Ofir Ben Moshe 315923151 #########################
# DQN: Cart Pole
import random
import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from box import box
from tensorboardX import SummaryWriter


def build_network(lr, net_arch):
    """
    Build NN

    :param lr: Learning rate
    :param net_arch: Network architecture
    :return: NN - Get state as input and return Q(s,a)
    """
    model = Sequential()
    model.add(Dense(units=64, activation='relu', input_dim=net_arch.state_dim, kernel_initializer='he_uniform'))
    model.add(Dense(units=32, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(units=32, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(units=24, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(units=24, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(units=net_arch.action_dim, activation='linear', kernel_initializer='he_uniform'))

    model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
    model.summary()
    return model


class ReplayBuffer:
    def __init__(self, capacity):
        """
        Replay Buffer use to store and sample random batches of experiences

        :param capacity: The number of experience batches to hold
        """
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def store(self, experience):
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.position % self.capacity] = experience
        self.position += 1

    def sample(self, batch_size):
        mini_batch = random.sample(self.memory, batch_size)
        states = np.array([exp[0] for exp in mini_batch])
        actions = np.array([exp[1] for exp in mini_batch])
        next_states = np.array([exp[2] for exp in mini_batch])
        rewards = np.array([exp[3] for exp in mini_batch])
        not_dones = np.array([exp[4] for exp in mini_batch])
        return states, actions, next_states, rewards, not_dones

    def __len__(self):
        return len(self.memory)


class DQN_Agent:
    def __init__(self, hp, env_dim):
        """
        The DQN agent class. use to store experience, choose action, decaying epsilon, update the online network
        and test the agent

        :param hp: Hyper-parameters
        :param env_dim: Environment dimensions
        """
        self.hp = hp
        self.net_arch = net_arch
        self.epsilon = self.hp.max_epsilon
        self.memory = ReplayBuffer(capacity=self.hp.capacity)
        self.online_net = build_network(lr=self.hp.lr, net_arch=self.net_arch)
        self.target_net = build_network(lr=self.hp.lr, net_arch=self.net_arch)
        self.target_net.set_weights(self.online_net.get_weights())

    def store_experience(self, experience):
        self.memory.store(experience=experience)

    def choose_action(self, observation):
        if random.uniform(0, 1) < self.epsilon:
            a = np.random.choice(self.net_arch.action_dim)  # Explore
        else:
            a = np.argmax(self.online_net.predict(x=np.array([observation])))  # Exploit
        self.epsilon_decay()
        return a

    def epsilon_decay(self):
        self.epsilon = max(self.hp.min_epsilon, self.epsilon * self.hp.epsilon_decay)

    def update_target_network(self):
        self.target_net.set_weights(self.online_net.get_weights())

    def update_step(self):
        self.epsilon_decay()
        if len(self.memory) < self.hp.batch_size: return 0

        # Sample minibatch
        states, actions, next_states, rewards, not_dones = self.memory.sample(batch_size=self.hp.batch_size)

        predicted_q = self.online_net.predict(states)  # Q(states_batch,:) from the online network
        target_q_next = self.target_net.predict(next_states)  # Q(next_states_batch,:) from the target network

        q_target = np.copy(predicted_q)  # Copy of Q(s,:) from the online network
        batch_index = np.arange(self.hp.batch_size)

        # Q target = Q(s,a) = r + gamma * MAX[Q_target_net(s',A')] * not_dones
        q_target[batch_index, actions] = rewards + self.hp.gamma * np.amax(target_q_next, axis=1) * not_dones

        # Update online network to give q values as output from the suit states inputs
        loss = self.online_net.train_on_batch(states, q_target)
        return loss

    def test_agent(self, env, visualize='True'):
        s = env.reset()
        total_reward = 0.0
        finish = False

        while not finish:
            if visualize:
                env.render()
            q_vals = self.online_net.predict(np.array([s]))
            a = np.argmax(q_vals)
            s, r, finish, _ = env.step(a)
            total_reward += reward
        print("Total reward in the test: %.2f" % total_reward)
        env.close()


if __name__ == '__main__':
    tf.compat.v1.disable_eager_execution()
    env = gym.make('CartPole-v1')
    writer = SummaryWriter()
    # Parameters
    env_dim = {
        'state_dim': env.observation_space.shape[0],
        'action_dim': env.action_space.n,
    }

    # This HP win after 431 Episodes
    best_hyper_parameters = {
        'lr': 0.0001,
        'batch_size': 128,
        'capacity': 10000,
        'gamma': 0.99,
        'max_epsilon': 0.9,
        'min_epsilon': 0.01,
        'epsilon_decay': 0.999,
        'target_update_period': 100
    }

    # Use a dot “.” to access members of dictionary
    env_dim = box.Box(env_dim)
    hp = box.Box(best_hyper_parameters)

    agent = DQN_Agent(hp=hp, env_dim=env_dim)  # Build an agent object
    max_episodes = 1000
    max_steps = 500
    max_score = 475.0
    total_steps = 0
    episode_loss = []
    average_score = []
    for episode in range(max_episodes):
        state = env.reset()
        done = False
        episode_score = 0
        for step in range(max_steps):
            total_steps += 1
            action = agent.choose_action(observation=state)
            next_state, reward, done, _ = env.step(action)
            episode_score += reward

            agent.store_experience((state, action, next_state, reward, not done))
            if done:
                average_score.append(episode_score)
                print("Episode: {} | Score: {}".format(episode + 1, episode_score))
                break

            state = next_state
            loss = agent.update_step()
            writer.add_scalar('Loss in each training step', loss, total_steps)
            episode_loss.append(loss)

            if (total_steps + 1) % hp.target_update_period == 0:
                agent.update_target_network()

        writer.add_scalar('Total reward per episode', episode_score, episode)
        writer.add_scalar('Mean reward in the last 100 episodes', np.mean(average_score[-100:]), episode)

        if (episode + 1) % 100 == 0:
            print("100 Episodes Average Score: {}".format(np.mean(average_score[-100:])))

        if np.mean(average_score[-100:]) >= max_score:
            print(
                "\nGreat!! "
                "You win after: {} Episodes\n"
                "Average reward in the last 100 episodes: {}".format(episode + 1, np.mean(average_score[-100:])))
            break
    writer.close()
    agent.test_agent(env=env)
