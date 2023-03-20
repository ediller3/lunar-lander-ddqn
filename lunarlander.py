# -*- coding: utf-8 -*-
# Work based on algorithm outlined by DeepMind in Mnih et al. (2015)
# https://arxiv.org/pdf/1509.06461.pdf

import gym
from agent import Agent
import torch
from collections import deque
import numpy as np
import base64, io

# For visualization
import matplotlib.pyplot as plt
import seaborn as sns
from gym.wrappers.monitoring import video_recorder
from IPython.display import HTML
from IPython import display 
import glob

def dqn(env, agent, n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    try:
        scores = []                        # list containing scores from each episode
        avg_scores = []
        scores_window = deque(maxlen=100)  # last 100 scores
        eps = eps_start                    # initialize epsilon
        for i_episode in range(1, n_episodes+1):
            state = env.reset()
            score = 0
            for t in range(max_t):
                action = agent.act(state, eps)
                next_state, reward, done, _ = env.step(action)
                agent.step(state, action, reward, next_state, done)
                state = next_state
                score += reward
                if done:
                    break 
            scores_window.append(score)       # save most recent score
            scores.append(score)              # save most recent score
            eps = max(eps_end, eps_decay*eps) # decrease epsilon
            avg_scores.append(np.mean(scores_window))
            
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
            if i_episode % 100 == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
                torch.save(agent.qnetwork_local.state_dict(), 'drive/MyDrive/ll_checkpoints/{:d}_checkpoint.pth'.format(i_episode))
            if np.mean(scores_window)>=230.0:
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
                torch.save(agent.qnetwork_local.state_dict(), 'drive/MyDrive/ll_checkpoints/{:d}_checkpoint.pth'.format(i_episode))
                break
                
        print('\nEnvironment was not solved after {:d} episodes.\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        torch.save(agent.qnetwork_local.state_dict(), 'drive/MyDrive/ll_checkpoints/{:d}_checkpoint.pth'.format(i_episode))
        return scores, avg_scores

    except KeyboardInterrupt:
        print('\nEnvironment was interrupted after {:d} episodes.\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        torch.save(agent.qnetwork_local.state_dict(), 'drive/MyDrive/ll_checkpoints/{:d}_checkpoint.pth'.format(i_episode))
        return scores, avg_scores


"""
========================================================
Visualization functions (plotting, video rendering, etc.)
========================================================
"""
def plot_scores(avg_scores):
    plt.figure(figsize=(10, 5))
    ax = sns.lineplot(data=avg_scores, linewidth=2.5, color='orange', linestyle='dashed')
    ax.set(xlabel='Episode', ylabel='Scores', title='Moving average scores')
    plt.show()

def plot_loss(agent):
    plt.figure(figsize=(10, 5))
    loss_list = agent.loss_list
    ax = sns.lineplot(data=loss_list, linewidth=1, color='black')
    ax.set(xlabel='Learning step', ylabel='Loss', title='Local vs. Target Network MSE Loss')
    plt.ylim(0, 1000)
    plt.show()

def show_video(env_name):
    mp4list = glob.glob('/content/*.mp4')
    if len(mp4list) > 0:
        mp4 = '/content/{}.mp4'.format(env_name)
        video = io.open(mp4, 'r+b').read()
        encoded = base64.b64encode(video)
        display.display(HTML(data='''<video alt="test" autoplay 
                loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded.decode('ascii'))))
    else:
        print("Could not find video.")
        
def make_video(agent, env_name, ep_num):
    env = gym.make(env_name)
    vid = video_recorder.VideoRecorder(env, path="/content/{}.mp4".format(env_name))
    agent.qnetwork_local.load_state_dict(torch.load('drive/MyDrive/ll_checkpoints/{}_checkpoint.pth'.format(ep_num)))
    state = env.reset()
    done = False
    while not done:
        frame = env.render(mode='rgb_array')
        vid.capture_frame()
        
        action = agent.act(state)

        state, reward, done, _ = env.step(action)        
    env.close()

def gen_traj(agent, env_name, ep_num):
    env = gym.make(env_name)
    agent.qnetwork_local.load_state_dict(torch.load('drive/MyDrive/ll_checkpoints/{}_checkpoint.pth'.format(ep_num)))
    state = env.reset()
    done = False
    x_pos = []
    y_pos = []
    while not done:        
        action = agent.act(state)
        state, reward, done, _ = env.step(action)  
        x_pos.append(state[0])   
        y_pos.append(state[1])       
    env.close()

    return [x_pos, y_pos]

def plot_all_traj(traj_groups, labels):
  plt.figure(figsize=(12, 8))
  colors = sns.color_palette("pastel", len(traj_groups))

  for i, traj_list in enumerate(traj_groups):
    for traj in traj_list:
      sns.lineplot(x=traj[0], y=traj[1], sort=False, color=colors[i])

    for traj in traj_list:
      plt.plot(traj[0][-1], traj[1][-1], marker='x', markersize=5, color='black')

  plt.xlim(-1, 1)
  plt.ylim(-0.25, 2)
  plt.xticks(fontsize=8)
  plt.yticks(fontsize=8)
  plt.xlabel('x pos.', fontsize=12)
  plt.ylabel('y pos.', fontsize=12)
  plt.title('Trajectories across training episodes')

  handles = [plt.Line2D([], [], color=colors[i], label=label) for i, label in enumerate(labels)]
  plt.legend(handles=handles, title= "# of training eps.")
  plt.show()

def main():
    env = gym.make('LunarLander-v2')
    ll_agent = Agent(state_size=8, action_size=4, seed=0)
    scores, avg_scores = dqn(env, ll_agent)
    plot_scores(avg_scores)

    """
    # Renders video animation of policy executed using model at certain episode checkpoint.
    make_video(agent, 'LunarLander-v2', 1200)
    show_video('LunarLander-v2')
    """

    """
    # Plots example trajectories at different episode checkpoints. 
    ep_nums = [100, 200, 600, 1200]
    traj_groups = [[gen_traj(agent, 'LunarLander-v2', ep) for x in range(1, 6)] for ep in ep_nums]
    """

if __name__ == '__main__':
    main()