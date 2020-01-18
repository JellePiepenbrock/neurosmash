
import torch.nn as nn

class WMController(nn.Module):
    def __init__(self):
        super(WMController, self).__init__()

        self.controllermodel = nn.Sequential(
            nn.Linear(64, 3),
            nn.Softmax(dim = 0)
        )
    def forward(self, x):
        x = torch.FloatTensor(x)
        out = self.controllermodel(x)  
        return out


import Neurosmash
import seaborn as sns
import matplotlib.pyplot as plt
from torch.optim import Adam
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F
import numpy as np
import torch
from torch.autograd import Variable
from torch.distributions import Categorical
import cma
import pickle
from sklearn.decomposition import PCA
ip		 = "127.0.0.1" # Ip address that the TCP/IP interface listens to
port	   = 13000	   # Port number that the TCP/IP interface listens to
size	   = 768		 # Please check the Updates section above for more details
timescale  = 30	 # Please check the Updates section above for more details


env = Neurosmash.Environment(timescale=timescale) # This is the main environment.

    
        
def select_action(state,policy):
        # Select an action (0 or 1) by running policy model and choosing based on the probabilities in state
        state = torch.from_numpy(state).type(torch.FloatTensor)
        # print(state)
        state = policy(state)
        # print(state)
        c = Categorical(state)
        action = c.sample()
        # print(action)

        return action


def main():
        '''
        TODO: If the algorithm takes too long to converge, tune the following:
        'maxiter': '100 + 150 * (N+3)**2 // popsize**0.5  #v maximum number of iterations',
        'popsize': '4+int(3*np.log(N))  # population size, AKA lambda, number of new solution per iteration',
        'timeout': 'inf  #v stop if timeout seconds are exceeded, the string "2.5 * 60**2" evaluates to 2 hours and 30 minutes',
        'tolconditioncov': '1e14  #v stop if the condition of the covariance matrix is above `tolconditioncov`', 
        'tolfacupx': '1e3  #v termination when step-size increases by tolfacupx (diverges). That is, the initial step-size was chosen far too small and better solutions were found far away from the initial solution x0', 
        'tolupsigma': '1e20  #v sigma/sigma0 > tolupsigma * max(eivenvals(C)**0.5) indicates "creeping behavior" with usually minor improvements', 
        'tolfun': '1e-11  #v termination criterion: tolerance in function value, quite useful', 
        'tolfunhist': '1e-12  #v termination criterion: tolerance in function value history', 
        'tolstagnation': 'int(100 + 100 * N**1.5 / popsize)  #v termination if no improvement over tolstagnation iterations', 
        'tolx': '1e-11  #v termination criterion: tolerance in x-changes'
        opts.set('optname':value)
        '''
        ctrl = WMController()
        opts = cma.CMAOptions()
        opts.set('ftarget','inf')
        es = cma.CMAEvolutionStrategy(195 * [1], 1)
        while not es.stop():
            #store solutions for each sample in population
            solutions = []
            #get the population (20 samples)
            xs = es.ask(number=20)
            #for each sample in the population
            num = 0
            for person in xs:
                num = num + 1
                print("Person " + str(num))
                #reset the environment
                end, reward, state = env.reset() 
                done = False
                total_reward = 500 #the process minimizes the function

                #set the parameters of the policy
                last_ind = 0
                keys = []
                values = []
                for name, param in ctrl.named_parameters():
                    size_of_params = param.size()
                    total_size = np.prod(list(size_of_params))
                    subsample = person[last_ind:last_ind+total_size]
                    subsample_tensor = torch.from_numpy(subsample).reshape(list(size_of_params))
                    last_ind = last_ind+total_size
                    keys.append(name)
                    values.append(subsample_tensor)
                state_dict = dict(zip(keys, values))
                ctrl.load_state_dict(state_dict)

                #run the game
                for time in range(500):
                    #TODO: Change the line below
                    state = state[0:64]
                    #select and execute an action according to the policy
                    action = select_action(np.array(state), ctrl).detach()
                    done, reward, state = env.step(action)
                    reward += time
                    if done:
                        break
                #append obtained reward to the solutions
                solutions.append(reward)
                print("Total reward: " + str(reward))
            print(solutions)
            #tell es what you got
            es.tell(xs,solutions)
            break 
        ## asigning best weights 
        best_solution = es.result[0]
        last_ind = 0
        keys = []
        values = []
        for name, param in ctrl.named_parameters():
            size_of_params = param.size()
            total_size = np.prod(list(size_of_params))
            subsample = best_solution[last_ind:last_ind+total_size]
            subsample_tensor = torch.from_numpy(subsample).reshape(list(size_of_params))
            last_ind = last_ind+total_size
            keys.append(name)
            values.append(subsample_tensor)
        state_dict = dict(zip(keys, values))
        ctrl.load_state_dict(state_dict)
        

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

