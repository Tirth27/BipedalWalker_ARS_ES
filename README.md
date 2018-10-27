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




