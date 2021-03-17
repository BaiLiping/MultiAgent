from tensorforce import Agent, Environment
import matplotlib.pyplot as plt
import numpy as np
import math
import pickle
from tqdm import tqdm
import gym
import statistics 



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
braket_size=100

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

length=np.zeros(episode_number)
measure_length=moving_average(length,average_over)
#reward_record_normal_average=pickle.load(open( "normal_average_record.p", "rb"))
normal_record=pickle.load(open("normal_record.p","rb"))
normal_average=[]
normal_standard_deviation=[]
for j in range(len(normal_record)-braket_size+1):
	braket_normal=normal_record[j:j+braket_size]
	normal_mean=statistics.mean(braket_normal)
	normal_average.append(normal_mean)
	normal_standard_deviation.append(statistics.stdev(braket_normal, xbar = normal_mean))
#reward_record_single_average=pickle.load(open( "single_average_record.p", "rb"))
single_record=pickle.load(open( "single_record.p", "rb"))
single_average=[]
single_standard_deviation=[]
for j in range(len(single_record)-braket_size+1):
	braket_single=single_record[j:j+braket_size]
	single_mean=statistics.mean(braket_single)
	single_average.append(single_mean)
	single_standard_deviation.append(statistics.stdev(braket_single, xbar = single_mean))

single_average=np.array(single_average)
single_standard_deviation=np.array(single_standard_deviation)

normal_average=np.array(normal_average)
normal_standard_deviation=np.array(normal_standard_deviation)

evaluation_reward_record_normal=pickle.load(open( "evaluation_normal_record.p", "rb"))
evaluation_reward_record_single=pickle.load(open( "evaluation_single_record.p", "rb"))
average_normal=sum(evaluation_reward_record_normal)/evaluation_episode_number
average_single=sum(evaluation_reward_record_single)/evaluation_episode_number


#plot
x=range(len(single_average))
plt.figure(figsize=(13,7))
plt.plot(x,normal_average,label='Normal Training \n Evaluation Average: \n %s' %average_normal,color='black')
plt.fill_between(x, normal_average-normal_standard_deviation, normal_average+normal_standard_deviation,color='gray',alpha=0.3)
plt.plot(x,single_average,label='MultiAgents Roll Out Training \n Evaluation Average: \n %s' %average_single,color='magenta')
plt.fill_between(x, single_average-single_standard_deviation, single_average+single_standard_deviation,color='magenta',alpha=0.3)
plt.xlabel('Episodes Number',fontsize='large')
plt.ylabel('Episode Reward',fontsize='large')
plt.legend(loc='upper left',ncol=1, borderaxespad=0,prop={'size': 12})
plt.savefig('Walker.png')
