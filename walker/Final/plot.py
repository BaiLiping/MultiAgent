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
episode_number=2000
evaluation_episode_number=10
average_over=100


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

length=np.zeros(episode_number)
measure_length=moving_average(length,average_over)
evaluation_reward_record_normal=pickle.load(open( "evaluation_normal_record.p", "rb"))
reward_record_normal_average=pickle.load(open( "normal_average_record.p", "rb"))

reward_record_single_average=pickle.load(open( "single_average_record.p", "rb"))
reward_record_single=pickle.load(open( "single_record.p", "rb"))
evaluation_reward_record_single=pickle.load(open( "evaluation_single_record.p", "rb"))

average_normal=sum(evaluation_reward_record_normal)/evaluation_episode_number
average_single=sum(evaluation_reward_record_single)/evaluation_episode_number


#plot
x=range(len(measure_length))
plt.figure(figsize=(13,7))
plt.plot(x,reward_record_normal_average,label='Normal Training \n Evaluation Average: \n %s' %average_normal,color='black')
plt.plot(x,reward_record_single_average,label='MultiAgents Roll Out Training \n Evaluation Average: \n %s' %average_single,color='magenta')
plt.xlabel('Episodes Number',fontsize='large')
plt.ylabel('Episode Reward',fontsize='large')
plt.legend(loc='upper left',ncol=1, borderaxespad=0,prop={'size': 12})
plt.savefig('Walker.png')
