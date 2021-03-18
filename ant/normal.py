from tensorforce import Agent, Environment
import matplotlib.pyplot as plt
import numpy as np
import math
import pickle
from tqdm import tqdm
import gym

episode_number=1000
average_over=50
evaluation_episode_number=10

def set_exploration(num_steps,initial_value,decay_rate,set_type='exponential'):
    exploration=dict(type=set_type, unit='timesteps',
                 num_steps=num_steps,initial_value=initial_value,
                 decay_rate=decay_rate)
    return exploration
exploration=set_exploration(100,0.9,0.5)

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

length=np.zeros(episode_number)
measure_length=moving_average(length,average_over)

if __name__ == "__main__":
    #Normal Agent
    environment_normal=Environment.create(environment='gym',level='Ant-v3')
    reward_record_normal=[]
    agent_normal = Agent.create(agent='agent.json', environment=environment_normal,exploration=exploration)
    print('training normal agent')
    for _ in tqdm(range(episode_number)):
        episode_reward=0
        states = environment_normal.reset()
        terminal= False
        while not terminal:
            actions = agent_normal.act(states=states)
            states, terminal, reward = environment_normal.execute(actions=actions)
            episode_reward+=reward
            agent_normal.observe(terminal=terminal, reward=reward)
        reward_record_normal.append(episode_reward)
        print(episode_reward)
    temp=np.array(reward_record_normal)
    reward_record_normal_average=moving_average(temp,average_over)
    pickle.dump(reward_record_normal_average, open( "normal_average_record.p", "wb"))
    pickle.dump(reward_record_normal, open( "normal_record.p", "wb"))

    #evaluate the normal agent
    episode_reward = 0.0
    evaluation_reward_record_normal=[]
    print('evaluating normal')
    for _ in tqdm(range(evaluation_episode_number)):
        episode_reward=0
        states = environment_normal.reset()
        internals = agent_normal.initial_internals()
        terminal = False
        while not terminal:
            actions, internals = agent_normal.act(
                states=states, internals=internals, independent=True, deterministic=True
            )
            states, terminal, reward = environment_normal.execute(actions=actions)
            episode_reward += reward
        evaluation_reward_record_normal.append(episode_reward)
        print(evaluation_reward_record_normal)
    pickle.dump(evaluation_reward_record_normal, open( "evaluation_normal_record.p", "wb"))
    agent_normal.close()
    environment_normal.close()