import gymnasium
from ale_py import ALEInterface
from ale_py.roms import SpaceInvaders  # type: ignore
    
def test_atari_random_actions() -> None:
    n_episode = 3
    ale = ALEInterface()
    ale.loadROM(SpaceInvaders)
    env = gymnasium.make("ALE/SpaceInvaders-v5", render_mode="human")
    total_rewards = []    
    for _ in range(n_episode):
        observation, info = env.reset()
        total_reward = 0
        is_done = False
        while not is_done:
            action = env.action_space.sample()
            observation, supports, reward, is_done, info = env.step(action)
            print(f'observation: {observation}, reward: {reward}, done: {is_done}, info: {str(info)}')
            total_reward += reward
            total_rewards.append(total_reward)
            if info['lives'] == 0:
                break
        
    print(f'Average total reward over {n_episode} episodes: {sum(total_rewards) / n_episode}')          