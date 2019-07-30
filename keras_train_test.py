# -*- coding: utf-8 -*-
from collections import deque, OrderedDict
import matplotlib.pyplot as plt
import gym
import numpy as np
import time

from keras_agent import KAgent, reset_seeds

from env_player import EnvPlayer

import itertools
import pandas as pd


def ddpg_td3(agent, env, n_episodes=5000, 
             max_t=1000, 
             train_every_steps=5,
             train_steps=0):
  print("Running DDPG outer training loop for {} episodes.".format(
      n_episodes,))
  print(" Training every {} steps for {} step(s)".format(
      train_every_steps, train_steps if train_steps != 0 else train_every_steps))
  scores_deque = deque(maxlen=100)
  steps_deque = deque(maxlen=100)
  avg_scores = []
  scores = []
  max_scores = []
  ep_times = []
  steps_per_ep = []
  best_model = None
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
                 train_every_steps=train_every_steps,
                 train_steps=train_steps)
      state = next_state
      score += reward
      if done:
          break 
    if train_every_steps == 0:
        agent.train(nr_iters=(t//5))
    scores_deque.append(score)
    scores.append(score)
    avg_scores.append(np.mean(scores_deque))
    if (max_scores == []) or (score > np.max(max_scores)):
      new_best = True      
    m100 = np.max(scores_deque)
    max_scores.append(m100)
    steps_deque.append(t)
    steps_per_ep.append(t)
    t_end = time.time()
    ep_time = t_end - t_start
    if t >= (0.5 * max_t):
      ep_times.append(ep_time)
    _cl = np.mean(agent.critic_losses)
    _al = np.mean(agent.actor_losses)
    print('\rEpisode {:>4}  Score/M100/Avg: {:>6.1f}/{:>6.1f}/{:>6.1f}  Steps: {:>3}  [μcLoss:{:>8.1e} μaLoss:{:>8.1e}]  t:{:>4.1f}s'.format(
        i_episode, score, m100, avg_scores[-1], t, _cl, _al, ep_time), end="")
    if new_best and (m100 > 0):
      label = "EP{:04}_MAXS_{:.0f}_AVG_{:.0f}_cl_{:.2f}_al_{:.2f}".format(
          i_episode, m100, avg_scores[-1], _cl, _al)
      best_model = agent.save_actor(label=label)      
    if i_episode % 100 == 0:
      mean_ep = np.mean(ep_times)
      elapsed = i_episode * mean_ep
      total = (n_episodes + 1) * mean_ep
      left_time_hrs = (total - elapsed) / 3600
      print('\rEpisode {:>4}  Score/M100/Avg: {:>6.1f}/{:>6.1f}/{:>6.1f}  AvStp: {:>3.0f}  [μcLoss:{:>8.1e} μaLoss:{:>8.1e}]  t-left: {:>4.1f} hr'.format(
          i_episode, score, m100, avg_scores[-1],  np.mean(steps_deque), _cl, _al, left_time_hrs))
      print("  Loaded steps: {:>10}".format(agent.step_counter))
      print("  Train iters:  {:>10}".format(agent.train_iters))
      print("  Actor update: {:>10}".format(agent.actor_updates))
  return scores, best_model, max_scores, avg_scores, steps_per_ep

if __name__ == '__main__':
  
  plt.style.use('ggplot')
  
  np.set_printoptions(precision=4)
  np.set_printoptions(linewidth=130)
  pd.set_option('display.max_rows', 500)
  pd.set_option('display.max_columns', 500)
  pd.set_option('display.width', 1000)     
  
  
  results = OrderedDict({
      "Model": [],
      "BestScore": [],
      "Top5":[]
      })
  
  full_grid = {
      
     "MAX_EP" : [2000],
      
     "METHOD" : ["DDPG-TD3", "DDPG", ],
     
     "TRAIN_EVERY" : [10],
     "TRAIN_STEPS" : [1, 0],
     
     "ACT_LNOIS" : [0],
     "ACT_WDEC"  : [0],
     "ACT_GRCL"  : [None],
     "ACT_BN"    : [True, False],
     "ACT_ACTIV" : ['relu'],
     "ACT_CINIT"  : [True],   #[True, False],
     
     "CRI_LNOIS" : [0],       #[0, 0.1],
     "CRI_WDEC"  : [1e-2, 0],
     "CRI_GRCL"  : [1],       #[1, None],
     "CRI_BN"    : [False],   #[False, True],   
     "CRI_S_BN"  : [True],    #[True, False],
     "CRI_ACTIV" : ["relu"],
     "CRI_CINIT"  : [True],    #[True, False],
     "CRI_SIMPL" : [False, True]
     
   }
  
  n_experiments = 0  

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

  _combs, params = grid_dict_to_values(params_grid=full_grid)
  n_grid = len(_combs)
  print("Full grid search options: {}".format(n_grid))
  
  if n_experiments > 0:  
    np_grid_pos = np.random.choice(n_grid, size=n_experiments, replace=False)  
    final_combs = [_combs[x] for x in np_grid_pos]
    n_selected_grid = len(final_combs)
  else:
    final_combs = _combs
    n_selected_grid = n_grid
    
  print("Selected grid search options: {}".format(n_selected_grid))
  
  
  for k in full_grid:
    results[k] = []  
  
  for i,grid_data in enumerate(final_combs):
    # first reset seeds
    reset_seeds()    
    
    gp = grid_pos_to_params(grid_data, params)
    for k in gp:
      print("  {:<12} {:>7}".format(k+":", str(gp[k])))
      
    n_episodes = gp['MAX_EP']
    
    _alg = gp['METHOD']
    _train_every_steps = gp["TRAIN_EVERY"]
    _train_steps = gp['TRAIN_STEPS']

    _actor_layer_noise = gp['ACT_LNOIS']
    _actor_reg = gp['ACT_WDEC']
    _actor_clip = gp['ACT_GRCL']
    _actor_bn = gp['ACT_BN']
    _actor_act = gp['ACT_ACTIV']
    _actor_init = gp['ACT_CINIT']

    _critic_layer_noise = gp['CRI_LNOIS']
    _critic_reg = gp['CRI_WDEC']
    _critic_clip = gp['CRI_GRCL']
    _critic_bn = gp['CRI_BN']
    _critic_state_bn = gp['CRI_S_BN']
    _critic_act = gp['CRI_ACTIV']
    _critic_init = gp['CRI_CINIT']
    _critic_simple = gp['CRI_SIMPL']

    
    TD3 = "TD3" in _alg
    
    alg_name = _alg+"_{:03}".format(i)
    print("Starting iter {}/{} {} ({})...".format(
        i+1, n_selected_grid, alg_name, ["{}={}".format(k,v) for k,v in gp.items()]))
    
    env = gym.make('BipedalWalker-v2')
    input_size, output_size = env.observation_space.shape[0], env.action_space.shape[0]
    print("State: {}  Action: {}".format(input_size, output_size))
  


    
    agent = KAgent(state_size=input_size, 
                   action_size=output_size, 
                   name=alg_name, 
                   TD3=TD3, 
                   
                   actor_layer_noise=_actor_layer_noise,
                   actor_layer_reg=_actor_reg,
                   actor_clip_norm=_actor_clip,
                   actor_batch_norm=_actor_bn,
                   actor_activation=_actor_act,
                   actor_custom_init=_actor_init,
                   
                   critic_layer_noise=_critic_layer_noise,
                   critic_layer_reg=_critic_reg,
                   critic_clip_norm=_critic_clip,
                   critic_batch_norm=_critic_bn,
                   critic_state_batch_norm = _critic_state_bn,
                   critic_activation=_critic_act,
                   critic_custom_init=_critic_init,
                   critic_simple=_critic_simple,
                   )
    
    
    res = ddpg_td3(agent=agent, env=env, n_episodes=n_episodes,
                   train_every_steps=_train_every_steps,
                   train_steps=_train_steps,
                   )
    scores, best_model, max_scores, avg_scores, steps_per_ep = res
    
    for k in gp:
      results[k].append(gp[k])
    results['BestScore'].append(max_scores[-1])
    results['Top5'].append(np.mean(max_scores[-5:]))
    results['Model'].append(alg_name)
  
    if best_model is not None:
      agent.load_actor(best_model)
      player = EnvPlayer(env=env, agent=agent, save_gif='test_{}_post_train.gif'.format(
                                                    alg_name))
      player.play(cont=False)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores)+1), scores, "-b", label='scores')
    plt.plot(np.arange(1, len(scores)+1), avg_scores, "-r", label='avg scores')
    plt.plot(np.arange(1, len(scores)+1), max_scores, "-g", label='max scores')
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.title(alg_name)
    plt.savefig(alg_name+'.png')
    plt.show()
  
    df = pd.DataFrame(results).sort_values("BestScore")
    print(df.iloc[:,:7])
    df.to_csv("results.csv")
      