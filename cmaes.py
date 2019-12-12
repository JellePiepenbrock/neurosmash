import Neurosmash
import seaborn as sns
import matplotlib.pyplot as plt
from torch.optim import Adam
from PIL import Image
import torch.nn.functional as F
import numpy as np
import torch
from torch.autograd import Variable
from torch.distributions import Categorical
import cma

ip		 = "127.0.0.1" # Ip address that the TCP/IP interface listens to
port	   = 13000	   # Port number that the TCP/IP interface listens to
size	   = 768		 # Please check the Updates section above for more details
timescale  = 10	 # Please check the Updates section above for more details

# agent = Neurosmash.Agent() # This is an example agent.
						   # It has a step function, which gets reward/state as arguments and returns an action.
						   # Right now, it always outputs a random action (3) regardless of reward/state.
						   # The real agent should output one of the following three actions:
						   # none (0), left (1) and right (2)

# agent = Neurosmash.RLAgent()
env = Neurosmash.Environment(timescale=timescale) # This is the main environment.
policy = Neurosmash.Policy()
optimizer = torch.optim.Adam(policy.parameters())

def select_action(state,policy):
	# Select an action (0 or 1) by running policy model and choosing based on the probabilities in state
	state = torch.from_numpy(state).type(torch.FloatTensor)
	# print(state)
	state = policy(state)
	#print(state)
	c = Categorical(state)
	action = c.sample()
	# print(action)

	# Add log probability of our chosen action to our history
	if policy.policy_history.dim() != 0:
		policy.policy_history = torch.cat([policy.policy_history, c.log_prob(action)])
	else:
		policy.policy_history = (c.log_prob(action))
	return action


def main():
	
	es = cma.CMAEvolutionStrategy(1215 * [1], 1)
	#es.popsize() = 5
	while not es.stop():
		#store solutions for each sample in population
		solutions = []
		#get the population (50 samples)
		xs = es.ask(number=5)
		#for each sample in the population
		for person in xs:
			#reset the environment
			end, reward, state = env.reset() 
			done = False
			total_reward = 100 #the process minimizes the function

			#set the parameters of the policy
			last_ind = 0
			keys = []
			values = []
			for name, param in policy.named_parameters():
				size_of_params = param.size()
				total_size = np.prod(list(size_of_params))
				subsample = person[last_ind:last_ind+total_size]
				subsample_tensor = torch.from_numpy(subsample).reshape(list(size_of_params))
				last_ind = last_ind+total_size
				keys.append(name)
				values.append(subsample_tensor)
			state_dict = dict(zip(keys, values))
			policy.load_state_dict(state_dict)

			#run the game
			for time in range(100):
				#select and execute an action according to the policy
				action = select_action(np.array(state), policy).detach()
				done, reward, state = env.step(action)
				if reward > 0:
					#if you actually get a reward, set it to maximum
					total_reward = 0
				else:
					#reward for existing
					total_reward = total_reward - 1
				if done:
					break
			#append obtained reward to the solutions
			solutions.append(total_reward)
		print(solutions)
		#tell es what you got
		es.tell(xs,solutions)
	
'''
1) create population
2) for each member of population
	- set the policy params
	- run the game
	- record the reward
3) tell
3) repeat until converges
'''

main()
