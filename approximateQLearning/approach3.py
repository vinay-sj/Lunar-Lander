import gym
import numpy as np
import matplotlib.pyplot as plt

class Lunar:
    def __init__(self, N,Gamma, Alpha, EpsilonStart, EpsilonEnd,actionSpace):
        self.weights = np.zeros(144)
        self.N = N
        self.GAMMA = Gamma
        self.ALPHA = Alpha
        self.epsStart = EpsilonStart
        self.epsEnd = EpsilonEnd
        self.ActionSpace = actionSpace

    def calculateFeatures(self, observation, action):
        features = np.zeros(144)
        for i in range(8):
            for j in range(4):
                features[i * 4 + j] = observation[i] if action == j else 0

        for i in range(7):
            for j in range(4):
                features[32 + i * 4 + j] = observation[i] * observation[i + 1] if action == j else 0

        for i in range(6):
            for j in range(4):
                features[60 + i * 4 + j] = observation[i] * observation[i + 2] if action == j else 0

        for i in range(5):
            for j in range(4):
                features[84 + i * 4 + j] = observation[i] * observation[i + 3] if action == j else 0

        for i in range(4):
            for j in range(4):
                features[104 + i * 4 + j] = observation[i] * observation[i + 4] if action == j else 0

        for i in range(3):
            for j in range(4):
                features[120 + i * 4 + j] = observation[i] * observation[i + 5] if action == j else 0

        for i in range(2):
            for j in range(4):
                features[132 + i * 4 + j] = observation[i] * observation[i + 6] if action == j else 0

        for i in range(1):
            for j in range(4):
                features[140 + i * 4 + j] = observation[i] * observation[i + 7] if action == j else 0

        return features

    def calculateQValue(self, observation, action):
        features = self.calculateFeatures(observation, action)
        q = 0
        for i in range(len(self.weights)):
            q += self.weights[i] * features[i]
        return q

    def actionWithMaxQValue(self, observation):
        maxq = -float("inf")
        maxAction = 0
        for action in self.ActionSpace:
            q = self.calculateQValue(observation, action)
            if (q > maxq):
                maxq = q
                maxAction = action
        return maxAction

    def epsilonDecay(self, step):
        self.a = self.epsStart
        self.b = np.log(self.epsStart / self.epsEnd) / (self.N * 1 / 2 - 1)
        return np.clip(self.a / (np.exp(self.b * step)), min(self.epsStart, self.epsEnd),
                       max(self.epsStart, self.epsEnd))

    def getAction(self, state, n):
        self.epsilon = self.epsilonDecay(n)
        if np.random.rand() > self.epsilonDecay(n):
            maxaction = self.actionWithMaxQValue(state)
        else:
            maxaction =  np.random.choice(self.ActionSpace)
        return maxaction

    def QValueforMaxAction(self, observation):
        q = self.calculateQValue(observation, self.actionWithMaxQValue(observation))
        return q

    def updateWeights(self, state, action, nextState, reward, n):
        diff = (reward + (self.GAMMA**n) * self.QValueforMaxAction(nextState)) - self.calculateQValue(state, action)
        features = self.calculateFeatures(state, action)
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] + (self.ALPHA * diff * features[i])


class Agent:
    def __init__(self, N, Gamma, Alpha, EpsilonStart, EpsilonEnd, shaping):
        self.env = gym.make('LunarLander-v2')
        self.N = N
        self.lunar = Lunar(N,Gamma, Alpha, EpsilonStart, EpsilonEnd,[i for i in range(self.env.action_space.n)])
        self.shaping = shaping

    def play_game(self):
        returns = []
        retwindow = 0
        for n in range(self.N):
            done = False
            length = 0
            ret = 0
            state = self.env.reset()
            while not done:
                if (n % 100 == 0):
                    self.env.render()
                action = self.lunar.getAction(state, n + length)
                nextState, reward, done, _ = self.env.step(action)
                ret = ret + reward

                if self.shaping:
                    reward = reward - 0.8 * (abs(nextState[4]))
                    reward = reward - 5 * (((nextState[1])) * (abs(nextState[3]) ** 2))
                    if (done and nextState[6] == 1 and nextState[7] == 1 and nextState[2] == 0.0 and nextState[
                        3] == 0.0 and abs(nextState[0]) < 1.0):
                        reward = 200

                self.lunar.updateWeights(state, action, nextState, reward, n)
                state = nextState
                length = length + 1
            retwindow += ret
            returns.append(ret)
            if (n % 100 == 0):
                print("Episode number:", n, " ret:", ret, " length of episode:", length)
                print("Avg return in last 100 episodes:", retwindow / 100)
                retwindow = 0
        self.plot(np.array(returns))

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
    episode_rewards = Agent(10000, 0.99, 0.001, 0.2, 0.1, False)
    episode_rewards.play_game()
