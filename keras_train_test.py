# -*- coding: utf-8 -*-
from collections import deque, OrderedDict
import matplotlib.pyplot as plt
import gym
import numpy as np
import time

from keras_agent import KAgent

from env_player import EnvPlayer

import itertools
import pandas as pd


def ddpg_td3(agent, env, n_episodes=5000, max_t=2000, 
             train_every_steps=5):
  print("Running DDPG outer training loop for {} episodes.".format(
      n_episodes,))
  scores_deque = deque(maxlen=100)
  steps_deque = deque(maxlen=100)
  scores = []
  max_scores = [0]
  ep_times = []
  for i_episode in range(1, n_episodes+1):
    t_start = time.time()
    state = env.reset()
    agent.reset()
    score = 0
    new_best = False
    for t in range(max_t):
      if agent.is_warming_up():
        action = env.action_space.sample()
      else:
        action = agent.act(state, add_noise=True)
      action = action.squeeze()
      next_state, reward, done, _ = env.step(action)
      agent.step(state, action, reward, next_state, done, 
                 train_every_steps=train_every_steps)
      state = next_state
      score += reward
      if done:
          break 
    if train_every_steps == 0:
        agent.train(nr_iters=(t//5))
    if score > max_scores[-1]:
      max_scores.append(score)
      new_best = True      
    scores_deque.append(score)
    scores.append(score)
    steps_deque.append(t)
    t_end = time.time()
    ep_time = t_end - t_start
    if t > (max_t*0.7): # record only long episodes
      ep_times.append(ep_time)
    _cl = np.mean(agent.critic_losses)
    _al = np.mean(agent.actor_losses)
    print('\rEpisode {:>4}  Score/Avg: {:>6.1f}/{:>6.1f}  Steps: {:>3}  [μcLoss:{:>8.1e} μaLoss:{:>8.1e}]  t:{:>4.1f}s'.format(
        i_episode, score, np.mean(scores_deque), t, _cl, _al, ep_time), end="")
    if new_best and score > -50:
      label = "EP{:04}_MAXS_{:.0f}_AVG_{:.0f}_cl_{:.2f}_al_{:.2f}".format(
          i_episode, max_scores[-1], np.mean(scores_deque), _cl, _al)
      best_model = agent.save_actor(label=label)      
    if i_episode % 50 == 0:
      mean_ep = np.mean(ep_times)
      elapsed = i_episode * mean_ep
      total = (n_episodes + 1) * mean_ep
      left_time_hrs = (total - elapsed) / 3600
      print('\rEpisode {:>4}  Score/Avg: {:>6.1f}/{:>6.1f}  AvStp: {:>3.0f}  [μcLoss:{:>8.1e} μaLoss:{:>8.1e}]  t-left: {:>4.1f} hr'.format(
          i_episode, score, np.mean(scores_deque),  np.mean(steps_deque), _cl, _al, left_time_hrs))
      print("  Loaded steps: {:>10}".format(agent.step_counter))
      print("  Train iters:  {:>10}".format(agent.train_iters))
      print("  Actor update: {:>10}".format(agent.actor_updates))
  return scores, best_model, max_scores

if __name__ == '__main__':
  
  n_episodes = 1000
  
  results = OrderedDict({
      "BestScore": [],
      "Top5":[]
      })
  
  full_grid = {
     "METHOD" : ["DDPG", "DDPG-TD3"],
     "STEP_TRAIN" : [10, 0],
     
     "ACTOR_LNO" : [0],
     "ACTOR_WD" : [1e-2, 0],
     "ACTOR_GCN" : [None],
     "ACTOR_BN" : [False],
     
     "CRITIC_LNO" : [0.1, 0],
     "CRITIC_WD" : [1e-2, 0],
     "CRITIC_GCN" : [1, None],
     "CRITIC_BN" : [True],
     
   }
  
  for k in full_grid:
    results[k] = []

  def grid_dict_to_values(params_grid):
    params = []
    values = []
    for k in params_grid:
      params.append(k)
      values.append(params_grid[k])
    combs = list(itertools.product(*values))
    return combs, params
  
  def grid_pos_to_params( grid_data, params):
    func_kwargs = {}
    for j,k in enumerate(params):
      func_kwargs[k] = grid_data[j]  
    return func_kwargs

  combs, params = grid_dict_to_values(params_grid=full_grid)
  n_full_grid = len(combs)

  for i,grid_data in enumerate(combs):
    gp = grid_pos_to_params(grid_data, params)
    for k in gp:
      print("  {:<12} {:>7}".format(k+":", str(gp[k])))
    
    _alg = gp['METHOD']
    _train_every_steps = gp["STEP_TRAIN"]

    _actor_layer_noise = gp['ACTOR_LNO']
    _actor_reg = gp['ACTOR_WD']
    _actor_clip = gp['ACTOR_GCN']
    _actor_bn = gp['ACTOR_BN']

    _critic_layer_noise = gp['CRITIC_LNO']
    _critic_reg = gp['CRITIC_WD']
    _critic_clip = gp['CRITIC_GCN']
    _critic_bn = gp['CRITIC_BN']

    
    TD3 = "TD3" in _alg
    
    alg_name = _alg 
    print("Starting iter {}/{} {} ({})...".format(
        i, n_full_grid, alg_name, ["{}={}".format(k,v) for k,v in gp.items()]))
    
    env = gym.make('BipedalWalker-v2')
    input_size, output_size = env.observation_space.shape[0], env.action_space.shape[0]
    print("State: {}  Action: {}".format(input_size, output_size))
  
    agent = KAgent(state_size=input_size, action_size=output_size, 
                   name=alg_name, TD3=TD3, 
                   
                   actor_layer_noise=_actor_layer_noise,
                   actor_layer_reg=_actor_reg,
                   actor_clip_norm=_actor_clip,
                   actor_batch_norm=_actor_bn,
                   
                   critic_layer_noise=_critic_layer_noise,
                   critic_layer_reg=_critic_reg,
                   critic_clip_norm=_critic_clip,
                   critic_batch_norm=_critic_bn,
                   )
    
    
    res = ddpg_td3(agent=agent, env=env, n_episodes=n_episodes,
                   train_every_steps=_train_every_steps,
                   )
    scores, best_model, max_scores = res
    
    for k in gp:
      results[k].append(gp[k])
    results['BestScore'].append(max_scores[-1])
    results['Top5'].append(np.mean(max_scores[-5:]))
  
    agent.load_actor(best_model)
    player = EnvPlayer(env=env, agent=agent, save_gif='test_{}_post_train.gif'.format(
                                                  alg_name))
    player.play(cont=False)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores)+1), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.title(alg_name)
    plt.savefig(alg_name+'.png')
    plt.show()
  
    df = pd.DataFrame(results).sort_values("BestScore")
    print(df)
    df.to_csv("results.csv")
      