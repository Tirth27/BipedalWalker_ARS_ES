# BipedalWalker_ARS_ES
BipedalWalker is an environment provided by the OpenAI [gym](https://gym.openai.com/envs/BipedalWalker-v2/). 
> Reward is given for moving forward, total 300+ points up to the far end. If the robot falls, it gets -100. Applying motor torque
> costs a small amount of points, more optimal agent will get better score. State consists of hull angle speed, angular velocity, 
> horizontal speed, vertical speed, position of joints and joints angular speed, legs contact with ground, and 10 lidar rangefinder
> measurements. There's no coordinates in the state vector.

## Problem
The problem is posed as a finite-horizon, non-deterministic Markov decision process (MDP), and is as interesting as it is difficult. The high dimensionality and continuous ranges of inputs (space) and outputs (actions) poses especially challenging examples of the lemmas of delayed reward, credit assignment, and exploration vs. exploitation. Moreover, while the MDP might guarantee convergence to a deterministic optimal policy in the limit, the dimensionality and continuous range poses the challenge that it cannot be enumerated in finite space complexity.

A successful learner will "solve" the reinforcement learning problem by achieving an average reward of 300 over 100 consecutive trials. The scope of the study will both optimize and examine the effect of hyperparameters, e.g. learning rates, discount, etc., on performance. Lastly, a holistic comparison of both the reinforcement learner and evolutionary learner will be provided.

## Algorithms
The chief algorithms to be employed include a synthesis of DQN and DDPG in an Actor-Critic ([Konda 2000](http://web.mit.edu/jnt/www/Papers/J094-03-kon-actors.pdf)) architecture with batch normalization ([Ioffe](https://arxiv.org/abs/1502.03167)), and an application of an evolutionary strategy ([ES](https://arxiv.org/abs/1703.03864)) for optimization as a surrogate for traditional reinforcement learning.

The reinforcement learning algorithm is well-defined in ([Lillicrap 2016](https://arxiv.org/pdf/1509.02971.pdf)), and borrows extensively from ([Mnih 2013](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)), using the concepts of experience replay and separate target networks:
```
randomly initialize critic network Q and actor μ
initialize target network Q' and μ'
initialize replay buffer R
for episode = 1, M do:
  initialize a random process N for action exploration
  receive initial observation state s1
  for t = 1, T do:
    select action a_t = μ + N_t according to current policy
    execute action a_t and observe reward r_t and new state s_t+1
    store experience in replay buffer
    sample a random minibatch of N transitions from R
    update target values according to discount, γ
    update the actor policy using the sampled policy gradient
    update the target networks      
```
The evolutionary strategy (ES) proposed in ([Salimans 2017](https://arxiv.org/abs/1703.03864)), in brief, applies randomized, "black box" optimization as an alternative to traditional reinforcement learning:
```
input: learning rate α, noise standard deviation σ, initial policy parameters θ_0
for episode = 1, M do:
  for t = 1, T do:
    sample ϵ_1 ... ϵ_n
    compute returns
    update policy parameters, θ_t+1

```
ARS is a random search method for training linear policies for continuous control problems, based on the paper ["Simple random search provides a competitive approach to reinforcement learning."](https://arxiv.org/abs/1803.07055)


## Dependencies

```
pip install gym
```
- [evostra](https://github.com/alirezamika/evostra)

## Augmented Random Search
The credits for this code go to [colinskow](https://github.com/colinskow/move37/tree/master/ars). I've merely created a wrapper.
He explian the ARS in his video I just added the comment in the [code](https://github.com/Tirth27/BipedalWalker_ARS_ES/blob/master/Augmented%20Random%20Search/Ars_BipedalWalker.py) for better understand. 

Detial for the ColinSkow ARS algorithm explaination is in [Move37](https://www.theschool.ai/courses/move-37-course/lessons/augmented-random-search-tutorial-teach-a-robot-to-walk/) course by [SirajRaval](https://github.com/llSourcell)


## Evolution Strategies
[ES](https://arxiv.org/abs/1703.03864) is somewhat unfortunately named, as I view it has more of a variation of Random Search than in any way being related to the principles of evolution. Before describing ES, I think it’s helpful to understand Random Search. Consider the problem of finding wights for a neural network in a Reinforcement Learning task, where the network is given a score based on how well it performs in a particular environment. Random Search samples points (potential weights) in a hyper-sphere around the origin. The ‘best’ point is selected and then a hyper-sphere around that best point is sampled. The next best point is found and the cycle repeats until some stopping criteria. ES is similar to Random Search but instead of the new ‘best’ point being decided by the single best performing point, the new ‘best’ is calculated using the average of the difference between the previous ‘best’ and the sampled points, weighted by their ‘fitness’ (the value returned from the environment). Pseudo code is below. In this way, the new ‘best’ will move in the direction that maximizes performance in the environment. This can be considered as a type of gradient descent based on finite differences, even though no gradient is being calculated.

For further reading regarding ES read [https://blog.openai.com/evolution-strategies/](https://blog.openai.com/evolution-strategies/)

Im trying to implement using the [alirezamikas](https://github.com/alirezamika/bipedal-es) [evostra](https://github.com/alirezamika/evostra) but don't know the reward was not increasing. I train the agent over 1000 iterations

![ALT test](https://github.com/Tirth27/BipedalWalker_ARS_ES/blob/master/ES_Inherrited/images/Screenshot%20from%202018-10-27%2020-33-50.png)

For the sake of simplicity I first implement ES on CartPole environment in [Es_CartPole.py](https://github.com/Tirth27/BipedalWalker_ARS_ES/blob/master/Evolution%20Strategies/Es_CartPole.py) then transfer that concept into BipedalWalker environment in [Es_BipedalWalker.py](https://github.com/Tirth27/BipedalWalker_ARS_ES/blob/master/Evolution%20Strategies/Es_BipedalWalker.py) but it don't work well.
I found the ES implementation in python by hardmaru in his [repo](https://github.com/hardmaru/estool) It is easy to use tool to implement various ES strategies

Then I used the ARS to train the BipedalWalker 
![ALT test](https://raw.githubusercontent.com/Tirth27/BipedalWalker_ARS_ES/master/Augmented%20Random%20Search/videos/BipedalWalker-v2/openaigym.video.0.6124.video023132.mp4)


