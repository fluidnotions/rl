"""
A simple, yet effective, approach is to map an observation to a vector of two numbers
representing two actions. The action with the higher value will be picked. The linear
mapping is depicted by a weight matrix whose size is 4 x 2 since the observations are 4-
dimensional in this case. In each episode, the weight is randomly generated and is used to
compute the action for every step in this episode. The total reward is then calculated. This
process repeats for many episodes and, in the end, the weight that enables the highest total
reward will become the learned policy. This approach is called random search because the
weight is randomly picked in each trial with the hope that the best weight will be found
with a large number of trials.
"""
import gymnasium as gym
# from ale_py import ALEInterface
# from ale_py.roms import SpaceInvaders  # type: ignore
import torch
import numpy as np

def setup():
    # ale = ALEInterface()
    # ale.loadROM(SpaceInvaders)
    env = gym.make("CartPole-v1", render_mode="human")
    return env

def run_episode(env, weight):
    """
    Here, we convert the state array to a tensor of the float type because we need to
    compute the multiplication of the state and weight
    tensor, torch.matmul(state, weight), for linear mapping. The action with
    the higher value is selected using the torch.argmax() operation. And don't
    forget to take the value of the resulting action tensor using .item() because it is
    a one-element tensor.
    
    `torch.matmul` is a function in the PyTorch library that performs matrix multiplication 
    between two tensors. It is typically used for linear algebra operations in deep learning models.
    """
    observation, info = env.reset()
    # print(f'observation: type: {type(observation)}, info: {str(info)}')
    total_reward = 0
    is_done = False
    while not is_done:  
        state = torch.from_numpy(observation).float()
        action = torch.argmax(torch.matmul(state, weight))
        observation, supports, reward, is_done, info = env.step(action.item())
        if reward:
            total_reward += 1
        return total_reward
    
def test_get_observation_space() -> None:
    """
    In the context of developing a reinforcement learning (RL) random search policy, an observation space of 'shape: (210, 160, 3)' 
    means that each observation is a 3-dimensional RGB image of size 210x160 pixels. 
    This observation represents the current state of the game and is provided as input to the RL agent's policy network. 
    The policy network then outputs an action based on this observation, which is taken by the agent in the environment. 
    A random search policy involves selecting actions randomly from the action space and evaluating their performance in the environment. 
    By iterating this process over many episodes, the policy can learn to improve its performance in the game.
    """
    env = setup()
    shape: tuple[int, int, int] | None = env.observation_space.shape
    if shape is None:
        raise ValueError('shape is None')
    print(f'shape: {shape}') # (4,)
    
def test_get_action_space():
    """
    Action space type: <class 'gymnasium.spaces.discrete.Discrete'>
    Number of actions: 2
    """
    env = setup()
    action_space = env.action_space
    print(f"Action space type: {type(action_space)}")
    if isinstance(action_space, gym.spaces.Discrete):
        print(f"Number of actions: {action_space.n}")
    elif isinstance(action_space, gym.spaces.Box):
        print(f"Action space shape: {action_space.shape}")
    else:
        print("Unsupported action space type")    
      

def test_random_search_policy() -> None:
    env = setup()
    n_episode = 1000
    best_total_reward = 0
    best_weight = None
    total_rewards = []
    n_state = env.observation_space.shape[0] # type: ignore
    if isinstance(env.action_space, gym.spaces.Discrete):
        n_action = env.action_space.n
        for episode in range(n_episode):
            weight = torch.rand(n_state, int(n_action))
            total_reward = run_episode(env, weight)
            print(f'Episode {episode+1}: {total_reward}')
            if total_reward is not None and total_reward > best_total_reward:
                best_weight = weight
                best_total_reward = total_reward
            total_rewards.append(total_reward)
        sum_total_rewards = sum(total_rewards)    
        print(f'Average total reward over {n_episode} episodes: {sum_total_rewards / n_episode}. sum_total_rewards: {sum_total_rewards}, best_total_reward: {best_total_reward}, best_weight: {best_weight}')   
    

    
    
           