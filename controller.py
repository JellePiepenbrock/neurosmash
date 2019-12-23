import torch.nn as nn
import torch
from torch.distributions import Categorical

class Controller(nn.Module):

    def __init__(self):
        super(Controller, self).__init__()

        # 32 latent vars from VAE, 100 hidden state vars from LSTM
        self.predictor = nn.Sequential(
            nn.Linear(332, 10),
            nn.ReLU(),
            nn.Linear(10, 3)
        )

    def forward(self, x):
        x = self.predictor(x)

        return torch.softmax(x, dim=1)


def select_action(state, policy):
    # Select an action (0 or 1) by running policy model and choosing based on the probabilities in state

    # print(state)
    preds = policy(state)
    # print(state)
    c = Categorical(preds)
    action = c.sample()

    # Add log probability of our chosen action to our history
    if policy.policy_history.dim() != 0:
        policy.policy_history = torch.cat([policy.policy_history, c.log_prob(action)])
    else:
        policy.policy_history = (c.log_prob(action))
    return action

def update_policy(policy):
    R = 0
    rewards = []

    # Reward discounting (i.e. smear reward back through history)
    for r in policy.reward_episode[::-1]:
        R = r + policy.gamma * R
        rewards.insert(0, R)

    # Scale rewards, just some autoscaling
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
