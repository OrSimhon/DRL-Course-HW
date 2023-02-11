######################### Or Simhon 315600486, Ofir Ben Moshe 315923151 #########################
# Q-learning: Frozen Lake
import numpy as np
import gym
import random
import seaborn as sb
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import product
from collections import OrderedDict
# from tensorboardX import SummaryWriter


class RunBuilder:
    @staticmethod
    def get_runs(params):
        Run = namedtuple('Run', params.keys())

        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))

        return runs


def getGLIE(num_episodes, max_epsilon, decay_factor, linear):
    """
    Get list of 'numOfEpisodes' epsilons decreasing along the learning process

    :param num_episodes: Number of episodes
    :param max_epsilon: Threshold for exploitation-exploitation ratio
    :param decay_factor: Decay factor
    :param linear: True for linear decay, else exponential decay
    :return: GLIE's parameters (epsilons)
    """

    minEpsilon = 0.001  # Minimal exploitation-exploitation ratio of 0.1%

    if linear:  # Calculate GLIE's epsilons using linear decay
        GLIE = [max(minEpsilon, max_epsilon - ((1-decay_factor) * i)) for i in range(num_episodes)]

    else:  # Calculate GLIE's epsilons using exponential decay
        GLIE = [max(minEpsilon, max_epsilon * (decay_factor ** i)) for i in range(num_episodes)]

    return GLIE


def plot_graphs(q, title, heat=False, **kwargs):
    """

    :param q: q values - 2nd array
    :param title: changeable title of the heat map
    :param heat: weather or not to plot only heatmap
    :param kwargs: rewards and steps
    :return:
    """
    figsize = (8, 6)
    # q_heat_map
    plt.figure(figsize=figsize)
    sb.heatmap(q, annot=True, cmap="Greens")
    plt.title(title, fontsize=20)
    plt.xlabel('Actions', fontsize=15)
    plt.ylabel('States', fontsize=15)

    if heat: return
    # Plot rewards per episode
    plt.figure(figsize=figsize)
    plt.plot(kwargs['rewards'])
    plt.title("Cumulative Reward Per Episode", fontsize=20)
    plt.xlabel('Episode number', fontsize=15)
    plt.ylabel('Reward', fontsize=15)

    # Plot mean rewards over the last 100 episodes
    plt.figure(figsize=figsize)
    plt.plot(np.arange(len(kwargs['steps'])) * 100, kwargs['avg_reward'])
    plt.title("Mean reward in the last 100 episodes", fontsize=20)
    plt.xlabel('Episode number', fontsize=15)
    plt.ylabel('Reward', fontsize=15)

    # Plot steps
    plt.figure(figsize=figsize)
    plt.plot(np.arange(len(kwargs['steps'])) * 100, kwargs['steps'])
    plt.title("Mean Steps to the goal", fontsize=20)
    plt.xlabel('Episode number', fontsize=15)
    plt.ylabel('Steps', fontsize=15)
    plt.tight_layout()
    plt.show()


def epsilon_greedy_action(env, q, eps):
    """
    Exploration-exploitation trade-off

    :param env: physics environment
    :param q: q function of the specific current state
    :param eps: threshold
    :return:
    """
    if random.uniform(0, 1) > eps:
        return np.argmax(q)
    else:
        return env.action_space.sample()


def Q_Learning(writer, lr, discount_factor, max_epsilon, decay_factor, linear, plot):
    """
    Choosing Hyper-Parameters for Q-Learning Algorithm

    :param lr:
    :param discount_factor:
    :param max_epsilon:
    :param decay_factor:
    :param writer:
    :return:
    """

    np.random.seed(1)  # Sets random seed to get reproducible results
    num_episodes = 5000
    max_steps_per_episode = 100
    totalRewardsPerEpisode = []  # Initialize list of rewards per episode
    av_steps2goal = []  # Initialize list of average steps number to goal
    avg_reward = []
    steps = []
    step = 0
    env = gym.make("FrozenLake-v1")  # Creating The Environment
    q_table = np.zeros((env.observation_space.n, env.action_space.n))  # Creating The Q-Table
    GLIE = getGLIE(num_episodes, max_epsilon, decay_factor, linear=linear)  # Get GLIE's epsilons

    # Q-Learning Algorithm Training Loop
    for episode in range(num_episodes):

        state = env.reset()  # Get agent's initial state (S0) using exploring starts
        done = False  # Terminal state flag
        cumulative_reward_episode = 0
        eps = GLIE[episode]  # Extract current episode epsilon

        for step in range(max_steps_per_episode):

            action = epsilon_greedy_action(env=env, q=q_table[state, :], eps=eps)  # Exploration-exploitation trade-off

            # Taking action and get from the environment the next state, reward and weather or not is it terminal state
            new_state, reward, done, info = env.step(action)

            # Update Q-Value(s,a)
            q_table[state, action] = q_table[state, action] * (1 - lr) + lr * (
                    reward + discount_factor * np.max(q_table[new_state, :]))

            # Transition to the next state
            state = new_state
            cumulative_reward_episode += reward

            if done:  # The agent reached to Hole or Goal step
                if reward < 1: step = 100  # Hole
                break
        steps.append(step)

        # append the rewards from the current episode to the list of rewards from all episodes
        # writer.add_scalar('reward_per_episode', cumulative_reward_episode, episode)
        totalRewardsPerEpisode.append(cumulative_reward_episode)

        if not (episode + 1) % 100:  # Every 100 episodes
            # writer.add_scalar('Mean Steps to the goal', np.mean(steps[-100:]), episode)
            # writer.add_scalar('Mean reward in the last 100 episodes', np.mean(totalRewardsPerEpisode[-100:]), episode)

            av_steps2goal.append(np.mean(steps[-100:]))
            avg_reward.append(np.mean(totalRewardsPerEpisode[-100:]))

        if episode in [499, 1999]: plot_graphs(q_table, title="Q-value table after {} episodes".format(episode + 1),
                                               heat=True)

    if plot: plot_graphs(q=q_table, title="Final Q-value table", rewards=totalRewardsPerEpisode, avg_reward=avg_reward,
                         steps=av_steps2goal)

    env.close()


def Optimize_Q_Learning():
    params = OrderedDict(
        learning_rate=[.1, .01, .001, .0001]
        , discount_factor=[0.9, 0.99, 0.999, 0.9995]
        , linear=[True, False]
        , decay_factor=[0.99, 0.999, 0.9995]
    )

    for run in RunBuilder.get_runs(params):
        writer = SummaryWriter(f'{run}')  # Initialize the SummaryWriter for tensorboard

        Q_Learning(writer=writer, lr=run.learning_rate, discount_factor=run.discount_factor,
                   max_epsilon=1, decay_factor=run.decay_factor, linear=run.linear, plot=False)

        writer.close()


if __name__ == "__main__":
    # Hyper-Parameters choosing using comparisons between different combinations in tensorboard graphs
    # Optimize_Q_Learning()

    # # Best Hyper-Parameters
    learning_rate = 0.1  # alpha
    discount_factor = 0.999  # gamma
    linear = True  # epsilon decay schedule
    decay_factor = 0.999  # epsilon decay factor

    # writer = SummaryWriter()

    Q_Learning(writer=None, lr=learning_rate, discount_factor=discount_factor,
               max_epsilon=1, decay_factor=decay_factor, linear=linear, plot=True)

    writer.close()
