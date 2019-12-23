import Neurosmash
import numpy as np
import torch
from VAE import VAE

ip         = "127.0.0.1" # Ip address that the TCP/IP interface listens to
port       = 13000       # Port number that the TCP/IP interface listens to
size       = 64         # Please check the Updates section above for more details
timescale  = 10    # Please check the Updates section above for more details
create_batches = False
batch_size = 128


# Load VAE weights
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vae = VAE(device, image_channels=3).to(device)
vae.load_state_dict(torch.load("vae_v2.torch"))

vae = vae.cuda()

def select_action():
    return np.random.randint(low=0, high=3)


def main(episodes, episode_length):
    env = Neurosmash.Environment(timescale=timescale, size=size, port=port, ip=ip) # This is the main environment.
    end, reward, state = env.reset()

    ep_cnt = 0
    all_states = []
    all_actions = []
    all_dones = []
    all_rewards = []

    file_cnt = 0

    while len(all_states) < episodes:
        episode_states = []
        episode_actions = []
        episode_dones = []
        episode_rewards = []

        end, reward, state = env.reset()  # Reset environment and record the starting state
        done = False
        ep_cnt += 1


        for time in range(episode_length):

            action = select_action()
            episode_actions.append(action)

            # Step through environment using chosen action
            done, reward, state = env.step(action)
            state = torch.FloatTensor(state).reshape(size, size, 3) / 255.0
            state = state.permute(2, 0, 1)

            episode_states.append(vae.encode(state.reshape(1, 3, 64, 64).cuda())[0])
            episode_actions.append(action)
            episode_dones.append(done)
            episode_rewards.append(reward)
            print(time)
            if done:
                break


        if len(episode_states) >= 20:
            all_states.append(torch.stack(episode_states)[-20:])
            all_actions.append(torch.Tensor(episode_actions[-20:]))
            all_dones.append(torch.Tensor(episode_dones[-20:]))
            all_rewards.append(torch.Tensor(episode_rewards[-20:]))

    print(episode_states[-1])
    print(episode_actions)
    print(episode_dones)
    print(episode_rewards)
    torch.save(torch.stack(all_states), 'training_data.pt')
    torch.save(torch.stack(all_actions), 'training_actions.pt')
    torch.save(torch.stack(all_dones), 'training_dones.pt')
    torch.save(torch.stack(all_rewards), 'training_rewards.pt')

main(100, 50)