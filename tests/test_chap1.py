import gymnasium
from ale_py import ALEInterface
from ale_py.roms import SpaceInvaders  # type: ignore
    
def test_gym_make():
    ale = ALEInterface()
    ale.loadROM(SpaceInvaders)
    env = gymnasium.make("ALE/SpaceInvaders-v5", render_mode="human")
    observation, info = env.reset()
    env.render()
    for _ in range(1000):
        action = env.action_space.sample()
        observation, supports, reward, done, info = env.step(action)
        print(f'observation: {observation}, supports: {supports}, reward: {reward}, done: {done}, info: {info}')
        env.render()
        if done:
            break