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
           x_position(exclude shown up in info instead) Not Limited
    0      rootz                                        Not Limited
    1      rooty                                        Not Limited
    2      thigh joint                                  -150           0
    3      leg joint                                    -150           0
    4      foot joint                                   -45           45
    5      velocity of rootx                           -10            10
    6      velocity of rootz                           -10            10
    7      velocity of rooty                           -10            10
    8      angular velocity of thigh joint             -10            10
    9      angular velocity of leg joint               -10            10
    10     angular velocity of foot joint              -10            10
    11     action for thigh joint
    12     action for leg joint
    13     action for foot joint

Actions:
    0     Thigh Joint Motor                             -1             1
    1     Leg Joint Motor                               -1             1
    2     Foot Joint Motor                              -1             1
Termination:
    healthy_angle_range=(-0.2, 0.2)
'''

#setparameters
num_steps=5000 #update exploration rate over n steps
initial_value=0.9 #initial exploartion rate
decay_rate=0.8 #exploration rate decay rate
set_type='exponential' #set the type of decay linear, exponential
exploration=dict(type=set_type, unit='timesteps',
                 num_steps=num_steps,initial_value=initial_value,
                 decay_rate=decay_rate)

episode_number=20000
evaluation_episode_number=10
average_over=50

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

length=np.zeros(episode_number)
measure_length=moving_average(length,average_over)

#Normal Agent
environment_normal=Environment.create(environment='gym',level='Hopper-v3')
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

#reward_record_normal_average=pickle.load(open( "normal_average_record.p", "rb"))

#training single action agent
environment_single = Environment.create(environment='gym', level='HopperBLP-v2')
reward_record_single=[]
agent_thigh = Agent.create(agent='agent.json', environment=environment_single,exploration=exploration)
agent_leg = Agent.create(agent='agent.json',environment=environment_single,exploration=exploration)
agent_foot = Agent.create(agent='agent.json',environment=environment_single,exploration=exploration)

print('training agent without single action')
for _ in tqdm(range(episode_number)):
    episode_reward=0
    states = environment_single.reset()
    terminal= False
    while not terminal:
        states[11]=0.0
        actions_thigh = agent_thigh.act(states=states)
        states[11]=actions_thigh[0]
        states[12]=0.0
        actions_leg = agent_leg.act(states=states)
        states[12]=actions_leg[0]
        states[13]=0.0
        actions_foot = agent_foot.act(states=states)
        states[13] = actions_foot[0]
        actions=[actions_thigh[0],actions_leg[0],actions_foot[0]]
        states, terminal, reward = environment_single.execute(actions=actions)
        episode_reward+=reward
        agent_thigh.observe(terminal=terminal, reward=reward)
        agent_leg.observe(terminal=terminal, reward=reward)
        agent_foot.observe(terminal=terminal, reward=reward)
    reward_record_single.append(episode_reward)
    print(episode_reward)
temp=np.array(reward_record_single)
reward_record_single_average=moving_average(temp,average_over)
pickle.dump(reward_record_single_average, open( "single_average_record.p", "wb"))
pickle.dump(reward_record_single, open( "single_record.p", "wb"))

#evaluate the gingle action agent in single action environment
episode_reward = 0.0
evaluation_reward_record_single=[]
print('evaluating single action agent')
for _ in tqdm(range(evaluation_episode_number)):
    episode_reward=0
    states = environment_single.reset()
    internals_thigh = agent_thigh.initial_internals()
    internals_leg = agent_leg.initial_internals()
    internals_foot = agent_foot.initial_internals()
    terminal = False
    while not terminal:
        actions_thigh, internals_thigh = agent_thigh.act(states=states, internals=internals_thigh, independent=True, deterministic=True)
        states[11]=actions_thigh[0]
        actions_leg, internals_leg = agent_leg.act(states=states, internals=internals_leg, independent=True, deterministic=True)
        states[12]=actions_leg[0]
        actions_foot, internals_foot = agent_foot.act(states=states, internals=internals_foot, independent=True, deterministic=True)
        states[13]=actions_foot[0]
        actions=[actions_thigh[0],actions_leg[0],actions_foot[0]]
        states, terminal, reward = environment_single.execute(actions=actions)
        episode_reward += reward
    evaluation_reward_record_single.append(episode_reward)
    print(evaluation_reward_record_single)
pickle.dump(evaluation_reward_record_single, open( "evaluation_single_record.p", "wb"))


#plot
x=range(len(measure_length))
plt.figure(figsize=(20,10))
plt.plot(x,reward_record_normal_average,label='normal agent',color='black')
plt.plot(x,reward_record_single_average,label='single agent',color='red')
plt.xlabel('episodes')
plt.ylabel('reward')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='center left',ncol=2,shadow=True, borderaxespad=0)
plt.savefig('plot_compare.png')



agent_thigh.close()
agent_leg.close()
agent_foot.close()
environment_single.close()
