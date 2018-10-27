import numpy as np
import gym

# Neural network to store state or  action policy 
# add hidden layer or node according to there need
IL = 4 # Input layer
HL = 20 #Hidden layer
OL = 2 #Output layer

w1 = np.random.randn(HL, IL) / np.sqrt(IL)
w2 = np.random.randn(OL, HL) / np.sqrt(HL)

NumWeights1 = len(w1.flatten())
NumWeights2 = len(w2.flatten())

# Forward Propagation
def predict(s, w1, w2):
    h = np.dot(w1, s) #Input to hidden layer
    h[h<0] = 0 #Relu
    out = np.dot(w2, h) #hidden layer to output
    out = 1.0 / (1 + np.exp(-out)) #Sigmoid
    return out

#load environment
env = gym.make("CartPole-v0")

#Parameter
NumEpisodes = 50
NumPolicies = 10
sigma = 0.1 #Sigma determines how far away from the original policy we will explore 
learning_rate = 0.001

Reward = np.zeros(NumPolicies)


# Start learning
for episode in range(NumEpisodes):
    #Generate random varations around original policy
    eps = np.random.randn(NumPolicies, NumWeights1 + NumWeights2)
    
    #evaluate each policy over one episode
    for policy in range(NumPolicies):
        
        w1_try = w1 + sigma * eps[policy, :NumWeights1].reshape(w1.shape)        
        w2_try = w2 + sigma * eps[policy, NumWeights1:].reshape(w2.shape)
    
    #inital state
    observation = env.reset() #observe inital state
    
    Reward[policy] = 0
    
    while True:
        env.render()
        Action = predict(observation, w1_try, w2_try)
        Action = np.argmax(Action)
        
        #execute action
        observation_new, reward, done, info = env.step(Action)
        
        #collect reward
        Reward[policy] += policy
        
        #update state
        observation = observation_new
        
        #end episode
        if done:
            break

#calculate incremental rewards
F = (Reward - np.mean(Reward))

#update weights of original policy according to rewards of all variations
weights_update = learning_rate/(NumPolicies * sigma) * np.dot(eps.T, F)

w1 += weights_update[:NumWeights1].reshape(w1.shape)
w2 += weights_update[NumWeights1:].reshape(w2.shape)
        

