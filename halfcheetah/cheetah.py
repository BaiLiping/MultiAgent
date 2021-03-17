from tensorforce import Agent, Environment
import matplotlib.pyplot as plt
import numpy as np
import math
import pickle
from tqdm import tqdm
import gym

'''
For detailed notes on how to interact with the Mujoco environment, please refer
to note https://bailiping.github.io/Mujoco/

Observation:

    Num    Observation                                 Min            Max
           rootx(_get_obs states from  root z)          Not Limited
    0      rootz                                        Not Limited
    1      rooty                                        Not Limited
    2      thigh joint                                 -150           0
    3      leg joint                                   -150           0
    4      foot joint                                  -45            45
    5      thigh left joint                            -150           0
    6      leg left joint                              -150           0
    7      foot left joint                             -45            45
    8      velocity of rootx                           -10            10
    9      velocity of rootz                           -10            10
    10     velocity of rooty                           -10            10
    11     angular velocity of thigh joint             -10            10
    12     angular velocity of leg joint               -10            10
    13     angular velocity of foot joint              -10            10
    14     angular velocity of thigh left joint        -10            10
    15     angular velocity of leg left joint          -10            10
    16     angular velocity of foot left joint         -10            10

Actions:
    0     Thigh Joint Motor                             -1             1
    1     Leg Joint Motor                               -1             1
    2     Foot Joint Motor                              -1             1
    3     Thigh Left Joint Motor                        -1             1
    4     Leg Left Joint Motor                          -1             1
    5     Foot Left Joint Motor                         -1             1
Termination:
        done = not (height > 0.8 and height < 2.0 and
                    ang > -1.0 and ang < 1.0)
'''

episode_number=1000
average_over=50
evaluation_episode_number=10

def set_exploration(num_steps,initial_value,decay_rate,set_type='exponential'):
    exploration=dict(type=set_type, unit='timesteps',
                 num_steps=num_steps,initial_value=initial_value,
                 decay_rate=decay_rate)
    return exploration
exploration=set_exploration(100,0.9,0.5)
#setparameters


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

length=np.zeros(episode_number)
measure_length=moving_average(length,average_over)


#Normal Agent
environment_normal=Environment.create(environment='gym',level='HalfCheetah-v3')
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
