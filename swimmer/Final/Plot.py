from tensorforce import Agent, Environment
import matplotlib.pyplot as plt
import numpy as np
import math
import pickle
from tqdm import tqdm
import gym
import statistics 

from Normal import episode_number
from Normal import average_over
from Normal import evaluation_episode_number
from Normal import measure_length


braket_size=50

normal_record=pickle.load(open("normal_record.p","rb"))
normal_average=[]
normal_standard_deviation=[]
for j in range(len(normal_record)-braket_size+1):
	braket_normal=normal_record[j:j+braket_size]
	normal_mean=statistics.mean(braket_normal)
	normal_average.append(normal_mean)
	normal_standard_deviation.append(statistics.stdev(braket_normal, xbar = normal_mean))

normal_average=np.array(normal_average)
normal_standard_deviation=np.array(normal_standard_deviation)

evaluation_reward_record_normal=pickle.load(open( "evaluation_normal_record.p", "rb"))
average_normal=sum(evaluation_reward_record_normal)/evaluation_episode_number


#plot
x=range(len(normal_average))
plt.figure(figsize=(13,7))
plt.plot(x,normal_average,label='MultiAgent Training \n Evaluation Average: \n %s' %average_normal,color='black')
plt.fill_between(x, normal_average-normal_standard_deviation, normal_average+normal_standard_deviation,color='gray',alpha=0.3)
plt.xlabel('Episodes Number',fontsize='large')
plt.ylabel('Episode Reward',fontsize='large')
plt.legend(loc='upper left',ncol=1, borderaxespad=0,prop={'size': 12})
plt.savefig('Hopper.png')

