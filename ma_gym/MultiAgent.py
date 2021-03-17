from tensorforce import Agent, Environment
import matplotlib.pyplot as plt
import numpy as np
import math
import pickle
from tqdm import tqdm
import gym

from Normal import episode_number
from Normal import average_over
from Normal import evaluation_episode_number

from Normal import exploration
from Normal import measure_length
from Normal import moving_average

if __name__ == "__main__":

    #training single action agent
    environment = Environment.create(environment='gym', level='HopperBLP-v2')
    reward_record_single=[]
    agent_thigh = Agent.create(agent='agent.json', environment=environment,exploration=exploration)
    agent_leg = Agent.create(agent='agent.json',environment=environment,exploration=exploration)
    agent_foot = Agent.create(agent='agent.json',environment=environment,exploration=exploration)

    print('Training MultiAgents')
    for _ in tqdm(range(episode_number)):
        episode_reward=0
        states = environment.reset()
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
            states, terminal, reward = environment.execute(actions=actions)
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
    print('Evaluating MultiAgents')
    for _ in tqdm(range(evaluation_episode_number)):
        episode_reward=0
        states = environment.reset()
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
            states, terminal, reward = environment.execute(actions=actions)
            episode_reward += reward
        evaluation_reward_record_single.append(episode_reward)
        print(evaluation_reward_record_single)
    pickle.dump(evaluation_reward_record_single, open( "evaluation_single_record.p", "wb"))
    agent_thigh.save(directory='Thigh', format='numpy')
    agent_leg.save(directory='Leg', format='numpy')
    agent_foot.save(directory='Foot', format='numpy')
    agent_thigh.close()
    agent_leg.close()
    agent_foot.close()
    environment.close()

