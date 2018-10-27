import numpy as np
import gym
import random

class Model(object):
    
    def __init__(self, POPULATION_SIZE = 20, SIGMA = 0.1, EPS_AVG = 1,
                 INITIAL_EXPLORATION = 1.0, FINAL_EXPLORATION = 0.0,
                 EXPLORATION_DEC_STEPS = 1000000):
        self.weights = [np.zeros(shape=(24, 16)), np.zeros(shape=(16, 16)),
                        np.zeros(shape=(16, 4))]
        self.POPULATION_SIZE = POPULATION_SIZE
        self.SIGMA = SIGMA
        self.EPS_AVG = EPS_AVG
        self.INITIAL_EXPLORATION = INITIAL_EXPLORATION
        self.FINAL_EXPLORATION = FINAL_EXPLORATION
        self.EXPLORATION_DEC_STEPS = EXPLORATION_DEC_STEPS
        self.env = gym.make('BipedalWalker-v2')
        self.exploration = self.INITIAL_EXPLORATION
        self.AGENT_HISTORY_LENGTH = 1
        
    def predict(self, inputs):
        out = np.expand_dims(inputs.flatten(), 0)
        out = out / np.linalg.norm(out)        
        for layers in self.weights:
            out = np.dot(out, layers)
        return out[0]
    
    def get_predicted_action(self, sequence):
        prediction = Model.predict(np.array(sequence))
        return prediction
    
    def get_population(self):
        population = []
        for i in range(self.POPULATION_SIZE):
            x = []
            for w in self.weights:
                x.append(np.random.randn(*w.shape))
            population.append(x)
        return population
    
    def get_weights_try(self, w, p):
        weights_try = []
        for index, i in enumerate(p):
            jittered = self.SIGMA * i
            weights_try.append(w[index] + jittered)
        return weights_try
    
    def get_rewards(self, population):
        rewards = []
        for p in population:
            weights_try = self.get_weights_try(self.weights, p)
            rewards.append(self.get_reward(weights_try))
        rewards = np.array(rewards)
        return rewards
    
    def get_reward(self, weights):
        total_reward = 0.0
        self.set_weights(weights)

        for episode in range(self.EPS_AVG):
            observation = self.env.reset()
            sequence = [observation]*self.AGENT_HISTORY_LENGTH
            done = False
            while not done:
                self.exploration = max(self.FINAL_EXPLORATION, self.exploration - self.INITIAL_EXPLORATION/self.EXPLORATION_DEC_STEPS)
                if random.random() < self.exploration:
                    action = self.env.action_space.sample()
                else:
                    action = self.get_predicted_action(sequence)
                observation, reward, done, _ = self.env.step(action)
                total_reward += reward
                sequence = sequence[1:]
                sequence.append(observation)

        return total_reward/self.EPS_AVG
    
    def set_weights(self, weights):
        self.weights = weights

if __name__ == '__main__':
    env = gym.make("BipedalWalker-v2")
    Model = Model()
    #parameter
    NumEpisodes = 5000
    SIGMA = 0.1 #Sigma determines how far away from the original policy we will explore 
    learning_rate = 0.01
    # Start learning
    for episode in range(NumEpisodes):
        #Generate random varations around original policy
        populations = Model.get_population()
    
        #evaluate each policy over one episode
        for p in populations:
            weights_try = Model.get_weights_try(Model.weights, p)
    
        #inital state
        observation = env.reset() #observe inital state
    
        reward = Model.get_rewards(populations)
        sequence = [observation]*Model.AGENT_HISTORY_LENGTH
        total_reward = 0
        
        while True: 
            env.render()
            Action = Model.get_predicted_action(sequence)
            #Action = np.argmax(Action)
        
            #execute action
            observation_new, reward, done, info = env.step(Action)
        
            #collect reward
            total_reward += reward
        
            #update state
            sequence = observation_new
            print(reward)
            #end episode
            if done:
                break



