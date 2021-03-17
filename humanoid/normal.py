from tensorforce import Agent, Environment
import matplotlib.pyplot as plt
import numpy as np
import math
import pickle
from tqdm import tqdm
import gym

#setparameters
num_steps=100 #update exploration rate over n steps
initial_value=0.9 #initial exploartion rate
decay_rate=0.5 #exploration rate decay rate
set_type='exponential' #set the type of decay linear, exponential
exploration=dict(type=set_type, unit='timesteps',
                 num_steps=num_steps,initial_value=initial_value,
                 decay_rate=decay_rate)

episode_number=1000
evaluation_episode_number=10
average_over=50

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

length=np.zeros(episode_number)
measure_length=moving_average(length,average_over)
#Normal Agent
environment_normal=Environment.create(environment='gym',level='Humanoid-v3')
reward_record_normal=[]
agent_normal = Agent.create(agent='agent.json', environment=environment_normal,exploration=exploration)
print('training normal agent')
for _ in tqdm(range(episode_number)):
    episode_reward=0
    states = environment_normal.reset()
    terminal= False
    while not terminal:
        actions = agent_normal.act(states=states)
        print(actions)
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
'''
reward_record_normal_average=pickle.load(open( "normal_average_record.p", "rb"))
reward_record_normal=pickle.load(open( "normal_record.p", "rb"))
evaluation_reward_record_normal=pickle.load(open( "evaluation_normal_record.p", "rb"))

#training single action agent
environment_single = Environment.create(environment='gym', level='AntBLP-v0')
reward_record_single=[]
agent_one = Agent.create(agent='agent.json', environment=environment_single,exploration=exploration)
agent_two = Agent.create(agent='agent.json',environment=environment_single,exploration=exploration)
agent_three = Agent.create(agent='agent.json',environment=environment_single,exploration=exploration)
agent_four = Agent.create(agent='agent.json', environment=environment_single,exploration=exploration)
agent_five = Agent.create(agent='agent.json',environment=environment_single,exploration=exploration)
agent_six = Agent.create(agent='agent.json',environment=environment_single,exploration=exploration)
agent_seven = Agent.create(agent='agent.json',environment=environment_single,exploration=exploration)
agent_eight = Agent.create(agent='agent.json',environment=environment_single,exploration=exploration)

print('training agent without single action')
for _ in tqdm(range(episode_number)):
    episode_reward=0
    states=environment_single.reset()
    #first_action=[0,0,0,0,0,0,0,0]
    #states=environment_single.execute(actions=first_action)
    terminal= False
    while not terminal:
        states[111]=0.0
        actions_one = agent_one.act(states=states)
        states[111]=actions_one[0]
        states[112]=0.0
        actions_two = agent_two.act(states=states)
        states[112]=actions_two[0]
        states[113]=0.0
        actions_three = agent_three.act(states=states)
        states[113] = actions_three[0]
        states[114]=0.0
        actions_four = agent_four.act(states=states)
        states[114]=actions_four[0]
        states[115]=0.0
        actions_five = agent_five.act(states=states)
        states[115]=actions_five[0]
        states[116]=0.0
        actions_six = agent_six.act(states=states)
        states[116] = actions_six[0]
        states[117]=0.0
        actions_seven = agent_seven.act(states=states)
        states[117]=actions_seven[0]
        states[118]=0.0
        actions_eight = agent_eight.act(states=states)
        states[118] = actions_eight[0]       
        actions=[actions_one[0],actions_two[0],actions_three[0],actions_four[0],actions_five[0],actions_six[0],actions_seven[0],actions_eight[0]]
        states, terminal, reward = environment_single.execute(actions=actions)
        episode_reward+=reward
        agent_one.observe(terminal=terminal, reward=reward)
        agent_two.observe(terminal=terminal, reward=reward)
        agent_three.observe(terminal=terminal, reward=reward)
        agent_four.observe(terminal=terminal, reward=reward)
        agent_five.observe(terminal=terminal, reward=reward)
        agent_six.observe(terminal=terminal, reward=reward)
        agent_seven.observe(terminal=terminal, reward=reward)
        agent_eight.observe(terminal=terminal, reward=reward)
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
    internals_one = agent_one.initial_internals()
    internals_two = agent_two.initial_internals()
    internals_three = agent_three.initial_internals()
    internals_four = agent_four.initial_internals()
    internals_five = agent_five.initial_internals()
    internals_six = agent_six.initial_internals()
    internals_seven = agent_seven.initial_internals()
    internals_eight = agent_eight.initial_internals()
    terminal = False
    while not terminal:
        states[111]=0.0
        actions_one, internals_one = agent_one.act(states=states, internals=internals_one, independent=True, deterministic=True)
        states[111]=actions_one[0]
        states[112]=0.0
        actions_two, internals_two = agent_two.act(states=states, internals=internals_two, independent=True, deterministic=True)
        states[112]=actions_two[0]
        states[113]=0.0
        actions_three, internals_three = agent_three.act(states=states, internals=internals_three, independent=True, deterministic=True)
        states[113]=actions_three[0]
        states[114]=0.0
        actions_four, internals_four = agent_four.act(states=states, internals=internals_four, independent=True, deterministic=True)
        states[114]=actions_four[0]
        states[115]=0.0
        actions_five, internals_five = agent_five.act(states=states, internals=internals_five, independent=True, deterministic=True)
        states[115]=actions_five[0]
        states[116]=0.0
        actions_six, internals_six = agent_six.act(states=states, internals=internals_six, independent=True, deterministic=True)
        states[116]=actions_six[0]
        states[117]=0.0
        actions_seven, internals_seven = agent_seven.act(states=states, internals=internals_seven, independent=True, deterministic=True)
        states[117]=actions_five[0]
        states[118]=0.0
        actions_eight, internals_eight = agent_eight.act(states=states, internals=internals_eight, independent=True, deterministic=True)
        states[118]=actions_six[0]        
        actions=[actions_one[0],actions_two[0],actions_three[0],actions_four[0],actions_five[0],actions_six[0],actions_seven[0],actions_eight[0]]
        states, terminal, reward = environment_single.execute(actions=actions)
        episode_reward += reward
    evaluation_reward_record_single.append(episode_reward)
    print(evaluation_reward_record_single)
pickle.dump(evaluation_reward_record_single, open( "evaluation_single_record.p", "wb"))
'''
