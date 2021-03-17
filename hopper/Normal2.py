from tensorforce import Agent, Environment
import matplotlib.pyplot as plt
import numpy as np
import math
import pickle
from tqdm import tqdm
import gym

episode_number=10000
evaluation_episode_number=10
average_over=50

def set_exploration(num_steps,initial_value,decay_rate,set_type='exponential'):
    exploration=dict(type=set_type, unit='timesteps',
                 num_steps=num_steps,initial_value=initial_value,
                 decay_rate=decay_rate)
    return exploration
exploration=set_exploration(5000,0.9,0.8)

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

length=np.zeros(episode_number)
measure_length=moving_average(length,average_over)

if __name__ == "__main__":
    agent_normal=Agent.load(directory='Hopper', format='numpy')
    #Normal Agent
    environment=Environment.create(environment='gym',level='Hopper-v3')
    reward_record_normal=[]
    print('Training Normal Agent from Hopper')
    for _ in tqdm(range(episode_number)):
        episode_reward=0
        states = environment.reset()
        terminal= False
        action_old=[0,0,0]
        while not terminal:
            actions = agent_normal.act(states=states)
            states, terminal, reward = environment.execute(actions=actions)
            if actions[0]==action_old[0] and actions[1]==action_old[1] and actions[2]==action_old[2]:
                reward-=10
            if states[5]==0:
                reward-=10
            action_old=actions
            episode_reward+=reward
            agent_normal.observe(terminal=terminal, reward=reward)
        reward_record_normal.append(episode_reward)
        print(episode_reward)
    temp=np.array(reward_record_normal)
    reward_record_normal_average=moving_average(temp,average_over)
    pickle.dump(reward_record_normal_average, open( "normal_average_record2.p", "wb"))
    pickle.dump(reward_record_normal, open( "normal_record2.p", "wb"))

    #evaluate the normal agent
    episode_reward = 0.0
    evaluation_reward_record_normal=[]
    print('evaluating normal')
    for _ in tqdm(range(evaluation_episode_number)):
        episode_reward=0
        states = environment.reset()
        internals = agent_normal.initial_internals()
        terminal = False
        while not terminal:
            actions, internals = agent_normal.act(
                states=states, internals=internals, independent=True, deterministic=True
            )
            states, terminal, reward = environment.execute(actions=actions)
            episode_reward += reward
        evaluation_reward_record_normal.append(episode_reward)
        print(evaluation_reward_record_normal)
    pickle.dump(evaluation_reward_record_normal, open( "evaluation_normal_record2.p", "wb"))
    agent_normal.save(directory='Hopper2', format='numpy')
    agent_normal.close()
    environment.close()