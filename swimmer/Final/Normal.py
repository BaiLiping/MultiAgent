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
    #Normal Agent
    #environment=Environment.create(environment='gym',level='TrafficJunction-v0')
    environment=gym.make('Combat-v0')
    states=environment.reset()
    n_agent=len(states)
    observation_shape=len(states[0])*n_agent
    observation_specs=dict(type='float', shape=(observation_shape,))
    action_specs=dict(type='int', shape=(n_agent,),num_values=10)
    max_episode_timesteps=300
    reward_record_normal=[]
    agent_normal = Agent.create(agent='agent.json', states=observation_specs,actions=action_specs,max_episode_timesteps=max_episode_timesteps,exploration=exploration)
    print('Training Normal Agent')
    for _ in tqdm(range(episode_number)):
        episode_reward=0
        states_n = environment.reset()
        terminal= False
        while not terminal:
            states=states_n[0]
            for i in range(n_agent-1):
                states.extend(states_n[i+1])
            actions = agent_normal.act(states=states)
            states_n, reward_n, terminal_n,info = environment.step(actions)
            reward=sum(reward_n)
            episode_reward+=reward
            if False in terminal_n:
                terminal=False
            else:
                terminal=True
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
        states_n = environment.reset()
        internals = agent_normal.initial_internals()
        terminal = False
        while not terminal:
            states=states_n[0]
            for i in range(n_agent-1):
                states.extend(states_n[i+1])
            actions, internals = agent_normal.act(
                states=states, internals=internals, independent=True, deterministic=True
            )
            states_n, terminal_n, reward_n,info = environment.step(actions)
            reward=sum(reward_n)
            if False in terminal_n:
                terminal=False
            else:
                terminal=True
            episode_reward += reward
        evaluation_reward_record_normal.append(episode_reward)
        print(evaluation_reward_record_normal)
    pickle.dump(evaluation_reward_record_normal, open( "evaluation_normal_record.p", "wb"))
    agent_normal.save(directory='TrafficJunction', format='numpy')
    agent_normal.close()
    environment.close()