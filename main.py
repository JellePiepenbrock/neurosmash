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

ip         = "127.0.0.1" # Ip address that the TCP/IP interface listens to
port       = 13000       # Port number that the TCP/IP interface listens to
size       = 768         # Please check the Updates section above for more details
timescale  = 30     # Please check the Updates section above for more details

# agent = Neurosmash.Agent() # This is an example agent.
                           # It has a step function, which gets reward/state as arguments and returns an action.
                           # Right now, it always outputs a random action (3) regardless of reward/state.
                           # The real agent should output one of the following three actions:
                           # none (0), left (1) and right (2)

# agent = Neurosmash.RLAgent()
env = Neurosmash.Environment(timescale=timescale) # This is the main environment.
end, reward, state = env.reset()
policy = Neurosmash.Policy()
# policy.load_state_dict(torch.load("weights_100episodes_reward_quick_victory"))
optimizer = torch.optim.Adam(policy.parameters())

def select_action(state,policy):
    # Select an action (0 or 1) by running policy model and choosing based on the probabilities in state
    state = torch.from_numpy(state).type(torch.FloatTensor)
    # print(state)
    state = policy(state)
    # print(state)
    c = Categorical(state)
    action = c.sample()
    # print(action)
    # Add log probability of our chosen action to our history
    if policy.policy_history.dim() != 0:
        policy.policy_history = torch.cat([policy.policy_history, c.log_prob(action)])
    else:
        policy.policy_history = (c.log_prob(action))
    return action


def update_policy():
    R = 0
    rewards = []

    # Discount future rewards back to the present using gamma
    for r in policy.reward_episode[::-1]:
        R = r + policy.gamma * R
        rewards.insert(0, R)

    # Scale rewards
    rewards = torch.FloatTensor(rewards)
    rewards = torch.FloatTensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
    # print(rewards)
    # Calculate loss
    loss = (torch.sum(torch.mul(policy.policy_history, Variable(rewards)).mul(-1), -1))

    # Update network weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Save and intialize episode history counters
    policy.loss_history.append(loss.item())
    policy.reward_history.append(np.sum(policy.reward_episode).item())
    policy.policy_history = Variable(torch.Tensor())
    policy.reward_episode = []


def main(episodes):

    for episode in range(episodes):
        print("Episode: ", episode)
        end, reward, state = env.reset()  # Reset environment and record the starting state
        done = False

        for time in range(15):
            state = np.array(state)
            action = select_action(state, policy).detach()
            # Step through environment using chosen action
            done, reward, state = env.step(action)
            # Save reward
            if reward > 0:
                print(reward)

            # reward += time
            policy.reward_episode.append(reward)

            if done:
                break
        print(reward)



        update_policy()

main(50)
torch.save(policy.state_dict(), "weights_100episodes_reward_quick_victory2")