# banana-hunter
A deep-reinforcement learning agent that loves bananas, trained using the [Deep Q-Networks (DQN)](https://www.nature.com/articles/nature14236?wm=book_wap_0005).

## Project Details:
Banana-Hunter is a deep-reinforcement learning agent designed for the Banana Collectors environment from the [Unity ML-Agents Toolkit](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md).

![The banana collector environment](https://github.com/cptanalatriste/banana-hunter/blob/master/img/environment.png?raw=true)

The state is represented via a vector of 37 elements, corresponding to the agent's perception of the objects (i.e. bananas) around him.  Our agent has four possible actions:

1. Move forward, represented by 0.
1. Move backwards, represented by 1.
1. Turn left, represented by 2.
1. Turn right, represented by 3.

We considered our agent has mastered the task when he reached an **average score of 13, over 100 episodes**.

## Getting Started
Before running your agent, be sure to accomplish this first:
1. Clone this repository.
1. Download the banana collector environment appropriate to your operating system (available [here](https://github.com/udacity/deep-reinforcement-learning/tree/master/p1_navigation)).
1. Place the environment file in the cloned repository folder.
1. Setup an appropriate Python environment. Instructions available [here.]
(https://github.com/udacity/deep-reinforcement-learning)

##  Instructions
You can start running and training the agent by exploring `Navigation.ipynb`. Also available in the repository:

* `banana_hunter.py` contains the agent code.
* `banana_manager.py` has the code for training the agent.
