# -*- coding: utf-8 -*-
from collections import deque
import matplotlib.pyplot as plt
import gym
import numpy as np
import time

from keras_agent import KAgent

from env_player import EnvPlayer


def ddpg(agent, env, n_episodes=3000, max_t=700):
  scores_deque = deque(maxlen=100)
  scores = []
  steps = []
  max_score = -np.Inf
  ep_times = []
  for i_episode in range(1, n_episodes+1):
    t_start = time.time()
    state = env.reset()
    agent.reset()
    score = 0
    new_best = False
    for t in range(max_t):
        action = agent.act(state, add_noise=True)
        action = action.squeeze()
        next_state, reward, done, _ = env.step(action)
        agent.step(state, action, reward, next_state, done)
        state = next_state
        score += reward
        if done:
            break 
    if score > max_score:
      max_score = score
      new_best = True      
    scores_deque.append(score)
    scores.append(score)
    steps.append(t)
    t_end = time.time()
    ep_time = t_end - t_start
    ep_times.append(ep_time)
    _cl = np.mean(agent.critic_losses)
    _al = np.mean(agent.actor_losses)
    print('\rEpisode {:>4}  Score/Avg: {:>6.1f}/{:>6.1f}  Steps:{:>3}  [μcLoss:{:>5.1f} μaLoss:{:>5.1f}]  t:{:>4.1f}s'.format(
        i_episode, score, np.mean(scores_deque), t, _cl, _al, ep_time), end="")
    if new_best and score > -50:
      agent.save_actor()
    if i_episode % 50 == 0:
      elapsed = np.sum(ep_times)
      mean_ep = np.mean(ep_times)
      total = n_episodes * mean_ep
      left_time_hrs = (total - elapsed) / 3600
      print('\rEpisode {:>4}  Score/Avg: {:>6.1f}/{:>6.1f}  Steps:{:>3}  [μcLoss:{:>5.1f} μaLoss:{:>5.1f}]  t-left: {:>4.1f} hrs'.format(
          i_episode, score, np.mean(scores_deque),  t, _cl, _al, left_time_hrs))
  return scores

if __name__ == '__main__':

  algs = ['DDPG-TD3', "DDPG"]  
  for alg in algs:
    TD3 = "TD3" in alg
    print("Starting {} (TD3={})...".format(alg, TD3))
    
    env = gym.make('BipedalWalker-v2')
    input_size, output_size = env.observation_space.shape[0], env.action_space.shape[0]
    print("State: {}  Action: {}".format(input_size, output_size))
  
    agent = KAgent(state_size=input_size, action_size=output_size, 
                   name=alg, TD3=TD3)
    
    
    scores = ddpg(agent=agent, env=env)
  
    player = EnvPlayer(env=env, agent=agent, save_gif='test_{}_post_train.gif'.format(alg))
    player.play(cont=False)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores)+1), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.title(alg)
    plt.savefig(alg+'.png')
    plt.show()
      