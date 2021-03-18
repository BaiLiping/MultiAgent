from tensorforce import Agent, Environment
import matplotlib.pyplot as plt
import numpy as np
import math
import pickle
from tqdm import tqdm
import gym

from normal import episode_number
from normal import average_over
from normal import evaluation_episode_number

from normal import exploration
from normal import measure_length
from normal import moving_average

if __name__ == "__main__":
    #training single action agent
    environment_single = Environment.create(environment='gym', level='SwimmerBLP-v0')
    reward_record_single=[]
    agent_first = Agent.create(agent='agent.json', environment=environment_single,exploration=exploration)
    agent_second = Agent.create(agent='agent.json',environment=environment_single,exploration=exploration)

    print('training agent without single action')
    for _ in tqdm(range(episode_number)):
        episode_reward=0
        states = environment_single.reset()
        terminal= False
        while not terminal:
            actions_first = agent_first.act(states=states)
            states[9]=actions_first[0]
            actions_second = agent_second.act(states=states)
            states[8]=actions_second[0]
            actions=[actions_second[0],actions_first[0]]
            states, terminal, reward = environment_single.execute(actions=actions)
            episode_reward+=reward
            agent_first.observe(terminal=terminal, reward=reward)
            agent_second.observe(terminal=terminal, reward=reward)
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
        internals_first = agent_first.initial_internals()
        internals_second = agent_second.initial_internals()
        terminal = False
        while not terminal:
            actions_first, internals_first = agent_first.act(states=states, internals=internals_first, independent=True, deterministic=True)
            states[9]=actions_first[0]
            actions_second, internals_second = agent_second.act(states=states, internals=internals_second, independent=True, deterministic=True)
            states[8]=actions_second[0]
            actions=[actions_second[0],actions_first[0]]
            states, terminal, reward = environment_single.execute(actions=actions)
            episode_reward += reward
        evaluation_reward_record_single.append(episode_reward)
        print(evaluation_reward_record_single)
    pickle.dump(evaluation_reward_record_single, open( "evaluation_single_record.p", "wb"))
