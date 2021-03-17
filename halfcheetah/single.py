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

from normal import episode_number
from normal import average_over
from normal import evaluation_episode_number

from normal import exploration
from normal import measure_length
from normal import moving_average

if __name__ == "__main__":

    #training single action agent
    environment_single = Environment.create(environment='gym', level='HalfCheetahBLP-v0')
    reward_record_single=[]
    agent_thigh = Agent.create(agent='agent.json', environment=environment_single,exploration=exploration)
    agent_leg = Agent.create(agent='agent.json',environment=environment_single,exploration=exploration)
    agent_foot = Agent.create(agent='agent.json',environment=environment_single,exploration=exploration)
    agent_thigh_left = Agent.create(agent='agent.json', environment=environment_single,exploration=exploration)
    agent_leg_left = Agent.create(agent='agent.json',environment=environment_single,exploration=exploration)
    agent_foot_left = Agent.create(agent='agent.json',environment=environment_single,exploration=exploration)

    print('training agent without single action')
    for _ in tqdm(range(episode_number)):
        episode_reward=0
        states = environment_single.reset()
        terminal= False
        while not terminal:
            states[17]=0.0
            actions_thigh = agent_thigh.act(states=states)
            states[17]=actions_thigh[0]
            states[18]=0.0
            actions_leg = agent_leg.act(states=states)
            states[18]=actions_leg[0]
            states[19]=0.0
            actions_foot = agent_foot.act(states=states)
            states[19] = actions_foot[0]
            states[20]=0.0
            actions_thigh_left = agent_thigh_left.act(states=states)
            states[20]=actions_thigh_left[0]
            states[21]=0.0
            actions_leg_left = agent_leg_left.act(states=states)
            states[21]=actions_leg_left[0]
            states[22]=0.0
            actions_foot_left = agent_foot_left.act(states=states)
            states[22] = actions_foot_left[0]
            actions=[actions_thigh[0],actions_leg[0],actions_foot[0],actions_thigh_left[0],actions_leg_left[0],actions_foot_left[0]]
            states, terminal, reward = environment_single.execute(actions=actions)
            episode_reward+=reward
            agent_thigh.observe(terminal=terminal, reward=reward)
            agent_leg.observe(terminal=terminal, reward=reward)
            agent_foot.observe(terminal=terminal, reward=reward)
            agent_thigh_left.observe(terminal=terminal, reward=reward)
            agent_leg_left.observe(terminal=terminal, reward=reward)
            agent_foot_left.observe(terminal=terminal, reward=reward)
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
        internals_thigh_left = agent_thigh_left.initial_internals()
        internals_leg_left = agent_leg_left.initial_internals()
        internals_foot_left = agent_foot_left.initial_internals()
        terminal = False
        while not terminal:
            actions_thigh, internals_thigh = agent_thigh.act(states=states, internals=internals_thigh, independent=True, deterministic=True)
            states[17]=actions_thigh[0]
            actions_leg, internals_leg = agent_leg.act(states=states, internals=internals_leg, independent=True, deterministic=True)
            states[18]=actions_leg[0]
            actions_foot, internals_foot = agent_foot.act(states=states, internals=internals_foot, independent=True, deterministic=True)
            states[19]=actions_foot[0]
            actions_thigh_left, internals_thigh_left = agent_thigh_left.act(states=states, internals=internals_thigh, independent=True, deterministic=True)
            states[20]=actions_thigh_left[0]
            actions_leg_left, internals_leg_left = agent_leg_left.act(states=states, internals=internals_leg, independent=True, deterministic=True)
            states[21]=actions_leg_left[0]
            actions_foot_left, internals_foot_left = agent_foot_left.act(states=states, internals=internals_foot, independent=True, deterministic=True)
            states[22]=actions_foot_left[0]
            actions=[actions_thigh[0],actions_leg[0],actions_foot[0],actions_thigh_left[0],actions_leg_left[0],actions_foot_left[0]]
            states, terminal, reward = environment_single.execute(actions=actions)

        evaluation_reward_record_single.append(episode_reward)
        print(evaluation_reward_record_single)
    pickle.dump(evaluation_reward_record_single, open( "evaluation_single_record.p", "wb"))

