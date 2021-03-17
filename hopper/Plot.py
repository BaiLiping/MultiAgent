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
from Normal import measure_length

evaluation_reward_record_Normal=pickle.load(open( "evaluation_normal_record.p", "rb"))
reward_record_Normal_average=pickle.load(open( "normal_average_record.p", "rb"))

reward_record_single=pickle.load(open( "single_record.p", "rb"))
evaluation_reward_record_single=pickle.load(open( "evaluation_single_record.p", "rb"))
reward_record_single_average=pickle.load(open('single_average_record.p','rb'))

average_Normal=sum(evaluation_reward_record_Normal)/evaluation_episode_number
average_single=sum(evaluation_reward_record_single)/evaluation_episode_number

#plot
x=range(len(measure_length))
plt.figure(figsize=(13,7))
plt.plot(x,reward_record_Normal_average,label='Normal Training \n Evaluation Average: \n %s' %average_Normal,color='black')
plt.plot(x,reward_record_single_average,label='MultiAgents Roll Out Training \n Evaluation Average: \n %s' %average_single,color='magenta')
plt.xlabel('Episodes Number',fontsize='large')
plt.ylabel('Episode Reward',fontsize='large')
plt.legend(loc='lower right',ncol=1, borderaxespad=0,prop={'size': 12})
plt.savefig('hopper.png')
