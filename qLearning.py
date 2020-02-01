import random
import matplotlib.pyplot as plt
import gym
import numpy as np


class Lunar:
    def __init__(self, N, Gamma, Alpha, EpsilonStart, EpsilonEnd, actionSpace):
        self.MAXSTATES = 10 ** 8
        self.GAMMA = Gamma
        self.ALPHA = Alpha
        self.epsStart = EpsilonStart
        self.epsEnd = EpsilonEnd
        self.actionSpace = actionSpace
        self.N = N
        self.bins = np.zeros((8, 9))
        self.create_bins()
        self.qValue = {}
        self.initialize_Q()

    def create_bins(self):
        self.bins[0] = np.linspace(-1.0, 1.0, 9)
        self.bins[1] = np.linspace(-1.0, 1.0, 9)
        self.bins[2] = np.linspace(-1.0, 1.0, 9)
        self.bins[3] = np.linspace(-1.0, 1.0, 9)
        self.bins[4] = np.linspace(-1.0, 1.0, 9)
        self.bins[5] = np.linspace(-1.0, 1.0, 9)
        self.bins[6] = np.linspace(-1.0, 1.0, 9)
        self.bins[7] = np.linspace(-1.0, 1.0, 9)

    def initialize_Q(self):
        all_states = self.get_all_states_as_string()
        for state in all_states:
            self.qValue[state] = {}
            for action in self.actionSpace:
                self.qValue[state][action] = 0

    def getStates(self, observation):
        state = self.translateStateToString(self.discretize(observation))
        return state

    def discretize(self, observation):
        state = np.zeros(8)
        for i in range(8):
            state[i] = np.digitize(observation[i], self.bins[i])
        return state

    def translateStateToString(self, state):
        string_state = ''.join(str(int(e)) for e in state)
        return string_state

    def get_all_states_as_string(self):
        states = []
        for i in range(self.MAXSTATES):
            states.append(str(i).zfill(8))
        return states

    def computeValueFromQValues(self, state):
        value = []
        for action in self.actionSpace:
            value.append(self.qValue[state][action])
        if len(value) == 0:
            return 0.0
        return max(value)

    def computeActionFromQValues(self, state):
        value = self.computeValueFromQValues(state)
        tempaction = []
        for action in self.actionSpace:
            if value == self.qValue[state][action]:
                tempaction.append(action)
        if len(tempaction) == 0:
            return 0
        return random.choice(tempaction)

    def epsilonDecay(self, step):
        self.a = self.epsStart
        self.b = np.log(self.epsStart / self.epsEnd) / (self.N * 1 / 2 - 1)
        return np.clip(self.a / (np.exp(self.b * step)), min(self.epsStart, self.epsEnd),
                       max(self.epsStart, self.epsEnd))

    def getAction(self, state, n):
        legalActions = self.actionSpace
        if random.random() < self.epsilonDecay(n):
            return random.choice(legalActions)
        return self.computeActionFromQValues(state)

    def update(self, state, action, nextState, reward):
        self.qValue[state][action] = (self.qValue[state][action]) + self.ALPHA * (
                    reward + self.GAMMA * self.computeValueFromQValues(nextState) - (self.qValue[state][action]))


class Agent:
    def __init__(self, N, Gamma, Alpha, EpsilonStart, EpsilonEnd,shaping):
        self.env = gym.make('LunarLander-v2')
        self.N = N
        self.lunar = Lunar(N, Gamma, Alpha, EpsilonStart, EpsilonEnd, [i for i in range(self.env.action_space.n)])
        self.shaping = shaping

    def play_Episode(self, n):
        observation = self.env.reset()
        done = False
        state = self.lunar.getStates(observation)
        total_reward = 0
        length = 0

        while not done:
            if(n%100==0):
                self.env.render()
            action = self.lunar.getAction(state, n + length)
            observation, reward, done, _ = self.env.step(action)
            total_reward += reward
            nextState = self.lunar.getStates(observation)
            if(self.shaping):
                reward = reward - 0.8 * (abs(observation[4]))
                reward = reward - 5 * (((observation[1])) * (abs(observation[3]) ** 2))
                if (done and observation[6] == 1 and observation[7] == 1 and observation[2] == 0.0 and observation[
                    3] == 0.0 and abs(observation[0]) < 1.0):
                    reward = 200
            self.lunar.update(state, action, nextState, reward)
            state = nextState
            length = length + 1
        return total_reward, length

    def play_many_Episodes(self):
        reward = []
        retwindow = []
        for n in range(self.N):
            episode_reward, length = self.play_Episode(n)
            retwindow += episode_reward
            if n % 100 == 0:
                print("Episode number:", n, " Reward:", episode_reward, " length of episode:", length)
                print("Avg return in last 100 episodes:", retwindow / 100)
                retwindow = 0
            reward.append(episode_reward)
        self.plot(np.array(reward))
        return reward

    def plot(self, returns):
        smooth_returns = self.rolling_average(returns, window_size=250000)
        episodes = range(len(returns))
        plt.plot(episodes, returns, label="returns")
        plt.plot(episodes, smooth_returns, label="smoothened returns")
        plt.xlabel('Episode')
        plt.ylabel('Return')
        plt.legend(loc='best')
        plt.show()

    def rolling_average(self, data, *, window_size):
        assert data.ndim == 1
        kernel = np.ones(window_size)
        smooth_data = np.convolve(data, kernel) / np.convolve(
            np.ones_like(data), kernel
        )
        return smooth_data[: -window_size + 1]


if __name__ == '__main__':
    agent = Agent(10000, 0.99, 0.001, 0.2, 0.1,False)
    episode_rewards = agent.play_many_Episodes()
