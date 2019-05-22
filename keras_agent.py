# -*- coding: utf-8 -*-

import numpy as np
import random
from collections import namedtuple, deque

from tensorflow.keras import layers, models, optimizers, initializers
import tensorflow.keras.backend as K
import tensorflow as tf



class KAgent:
  """
  implementation of continuous state env agent based on
  Deterministic Deep Policy Gradient (DDPG) and TD3
  
  https://arxiv.org/pdf/1802.09477.pdf
  https://arxiv.org/pdf/1509.02971.pdf
  
  """
  def __init__(self, state_size, action_size, BUFFER_SIZE=int(1e6), BATCH_SIZE=128,
               env=None, random_seed=1234, GAMMA=0.99, TAU=5e-3, TD3=True, 
               policy_update_freq=2, noise_clip=0.5, 
               policy_noise=0.2, explor_noise=0.1,
               name='agent'):
    self.BUFFER_SIZE = BUFFER_SIZE
    self.BATCH_SIZE = BATCH_SIZE
    self.GAMMA = GAMMA
    self.TAU = TAU
    self.name = name
    self.state_size = state_size
    self.action_size = action_size
    self.actor_online = self._define_actor_model(state_size, action_size)
    self.actor_target = self._define_actor_model(state_size, action_size)
    _co, _cof = self._define_critic_model(state_size, action_size, 
                                          output_size=1, 
                                          compile_model=True)
    self.critic_online_1 = _co
    self.critic_online_frozen = _cof
    self.critic_target_1, _ = self._define_critic_model(state_size, action_size, 
                                                        output_size=1, 
                                                        compile_model=False)
    
    
    self.critic_online_2, _ = self._define_critic_model(state_size, action_size, 
                                                        output_size=1, 
                                                        compile_model=True)
    self.critic_target_2, _ = self._define_critic_model(state_size, action_size, 
                                                        output_size=1, 
                                                        compile_model=False)
    self.TD3 = TD3
    self.env = env
    
    if self.TD3:
      self.policy_update_freq = policy_update_freq
      self.policy_noise = policy_noise
      self.noise_clip = noise_clip
      self.explor_noise = explor_noise
      
    self._define_actor_trainer()
    
    self.max_action = 1 if self.env is None else self.env.action_space.high.max()
    self.min_action = -1 if self.env is None else self.env.action_space.low.min()
    
    # Noise process
    self.noise = OUNoise(action_size, random_seed)

    # Replay memory
    self.memory = ReplayBuffer(action_size, self.BUFFER_SIZE, 
                               BATCH_SIZE, random_seed)
    self.critic_losses = []
    self.actor_losses = []
    self.updates = 0
    self.actor_updates = 0
    print("Agent '{}' initialized with following params:\n {}".format(
        self.name, self.get_str()))
    return


  def get_str(self):
    obj = self
    out_str = obj.__class__.__name__+"("
    for prop, value in vars(obj).items():
      if type(value) in [int, float, bool]:
        out_str += prop+'='+str(value) + ','
      elif type(value) in [str]:
        out_str += prop+"='" + value + "',"
    out_str = out_str[:-1] if out_str[-1]==',' else out_str
    out_str += ')'
    return out_str
  
  
  def reset(self):
    self.noise.reset()


  def step(self, state, action, reward, next_state, done):
    """Save experience in replay memory, and use random sample from buffer to learn."""
    # Save experience / reward
    self.memory.add(state, action, reward, next_state, done)

    # Learn, if enough samples are available in memory
    if len(self.memory) > self.BATCH_SIZE:
      experiences = self.memory.sample()
      self.train(experiences, self.GAMMA)


  def act(self, state, add_noise=False):
    """Returns actions for given state as per current policy."""
    if len(state.shape) == 1:
      state = state.reshape((1,-1))
    action = self.actor_online.predict(state)
    if add_noise:
      if self.TD3:
        noise = np.random.normal(loc=0, scale=self.explor_noise, 
                                 size=action.shape)
      else:
        noise = self.noise.sample()
      action += noise
    return np.clip(action, self.min_action, self.max_action)


  def train(self, experiences, gamma):
    """Update policy and value parameters using given batch of experience tuples.
    Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
    where:
        actor_target(state) -> action
        critic_target(state, action) -> Q-value

    Params
    ======
        experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
        gamma (float): discount factor
    """
    self.updates += 1 # increment update
    
    states, actions, rewards, next_states, dones = experiences

    # ---------------------------- update critic ---------------------------- #
    # Get predicted next-state actions and Q values from target models
    actions_next = self.actor_target.predict(next_states)
    
    if self.TD3:
      noise = np.random.normal(loc=0.0, scale=self.policy_noise,
                               size=actions_next.shape)
      noise = np.clip(noise, -self.noise_clip, self.noise_clip)
      actions_next += noise
      actions_next = np.clip(actions_next, self.min_action, self.max_action)
    
    Q_targets_next_1 = self.critic_target_1.predict([next_states, actions_next])
    
    if self.TD3:
      Q_targets_next_2 = self.critic_target_2.predict([next_states, actions_next])    
      Q_targets_next = np.minimum(Q_targets_next_1, Q_targets_next_2)
    else:
      Q_targets_next = Q_targets_next_1
    
    # Compute Q targets for current states (y_i)
    Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))    
    
    # Train critic online
    critic_loss_1 = self.critic_online_1.train_on_batch(x=[states, actions], 
                                                        y=Q_targets)
    if self.TD3:
      critic_loss_2 = self.critic_online_2.train_on_batch(x=[states, actions], 
                                                          y=Q_targets)
      critic_loss = critic_loss_1 + critic_loss_2
    else:
      critic_loss = critic_loss_1
      
    self.critic_losses.append(critic_loss)
    
    if self.TD3 and (self.updates % self.policy_update_freq) != 0:
      # not yet update so skip the policy update and the targets update
      return
    
    # ---------------------------- update actor ---------------------------- #
    self.actor_updates += 1
    # now update actor after critic update
    actor_trainer_loss = self.actor_trainer.train_on_batch(x=states,y=None)
    self.actor_losses.append(actor_trainer_loss)

    # ----------------------- update target networks ----------------------- #
    self.soft_update(target=self.critic_target_1, source=self.critic_online_1, tau=self.TAU)
    self.soft_update(target=self.critic_target_2, source=self.critic_online_2, tau=self.TAU)
    self.soft_update(target=self.actor_target,  source=self.actor_online,  tau=self.TAU)                     
    return


  def save_actor(self):
    fn = '{}_upd_{}_{}.policy'.format(self.name, self.updates, self.actor_updates)
    self.actor_online.save(fn)


  def load_actor(self, fn):
    self.actor_online = tf.keras.models.load_model(fn)

  ###
  ###
  ###
  
  def _define_actor_trainer(self, lr=1e-4):
    tf_input = layers.Input((self.state_size,), name='actor_trainer_input')
    tf_actions_pred = self.actor_online(tf_input)
    # next line forces dQ / da so we can then propagate this grad in online actor
    tf_critic_values = self.critic_online_frozen([tf_input, tf_actions_pred])
    def actor_loss(y_true, y_pred):
      tf_loss = -K.mean(y_pred)
      return tf_loss
    opt = optimizers.Adam(lr=lr)
    self.actor_trainer = models.Model(inputs=tf_input, outputs=tf_critic_values)
    self.actor_trainer.compile(loss=actor_loss, optimizer=opt)
    
    
  
  def _define_actor_model(self, input_size, output_size):
    if not isinstance(input_size,tuple):
      input_size = (input_size,)
    tf_input = layers.Input(input_size, name='actor_input')
    tf_x = tf_input
    tf_x = layers.Dense(128, activation='relu',name='actor_dense_relu1')(tf_x)
    tf_x = layers.Dense(64, activation='relu',name='actor_dense_relu2')(tf_x)
    tf_x = layers.Dense(output_size, activation='tanh',name='actor_lin_tanh_out',
                        kernel_initializer=initializers.RandomUniform(-3e-3,3e-3))(tf_x)
    model = models.Model(inputs=tf_input, outputs=tf_x, name='actor')
    return model
    
  def _define_critic_model(self, input_size, action_size, output_size, 
                           act=None, compile_model=True, lr=1e-3):
    if not isinstance(input_size,tuple):
      input_size = (input_size,)
    if not isinstance(action_size, tuple):
      action_size = (action_size,)
    tf_input_state = layers.Input(input_size, name='critic_input_state')
    tf_input_action = layers.Input(action_size, name='critic_input_action')
    tf_x = tf_input_state
    tf_x = layers.Dense(128, activation='relu',name='critic_dense_relu1')(tf_x)
    tf_x = layers.concatenate([tf_x, tf_input_action], name='critic_concat_action_state')
    tf_x = layers.Dense(128, activation='relu',name='critic_dense_relu2')(tf_x)
    tf_x = layers.Dense(64, activation='relu',name='critic_dense_relu3')(tf_x)
    tf_x = layers.Dense(output_size, activation=act, name='critic_dense_out',
                        kernel_initializer=initializers.RandomUniform(-3e-3,3e-3))(tf_x)
    model = models.Model(inputs=[tf_input_state,tf_input_action], outputs=tf_x,
                         name='critic')
    
    if compile_model:
      opt = optimizers.Adam(lr=lr)    
      model.compile(loss='mse', optimizer=opt)
    
    model_frozen = models.Model(inputs=[tf_input_state,tf_input_action], outputs=tf_x,
                         name='frozen_critic')
    model_frozen.trainable = False
    return model, model_frozen
  
  def soft_update(self, target, source, tau):
    wt = target.get_weights()
    ws = source.get_weights()
    wf = [wt[i] * (1 - tau) + ws[i] * tau for i in range(len(wt))]
    target.set_weights(wf)
    return
  
  def soft_copy_actor(self, tau=0.005):
    self.soft_copy(self.actor_target, self.actor_online, tau=tau)
    
  def soft_copy_critic(self, tau=0.005):
    self.soft_copy(self.critic_target, self.critic_online, tau=tau)



class OUNoise:
  """Ornstein-Uhlenbeck process."""

  def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2, dt=1e-2):
    """Initialize parameters and noise process."""
    self.mu = mu * np.ones(size)
    self.theta = theta
    self.sigma = sigma
    self.dt = dt
    self.seed = random.seed(seed)
    self.reset()

  def reset(self):
    """Reset the internal state (= noise) to mean (mu)."""
    self.state = self.mu.copy()

  def sample(self):
    """Update internal state and return it as a noise sample."""
    x = self.state
    dx = self.theta * (self.mu - x) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.randn(len(x))
    self.state = x + dx
    return self.state




class ReplayBuffer:
  """Fixed-size buffer to store experience tuples."""

  def __init__(self, action_size, buffer_size, batch_size, seed):
    """Initialize a ReplayBuffer object.
    Params
    ======
        buffer_size (int): maximum size of buffer
        batch_size (int): size of each training batch
    """
    self.action_size = action_size
    self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
    self.batch_size = batch_size
    self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
    self.seed = random.seed(seed)
  
  def add(self, state, action, reward, next_state, done):
    """Add a new experience to memory."""
    e = self.experience(state, action, reward, next_state, done)
    self.memory.append(e)
  
  def sample(self):
    """Randomly sample a batch of experiences from memory."""
    experiences = random.sample(self.memory, k=self.batch_size)

    states = np.vstack([e.state for e in experiences if e is not None])
    actions = np.vstack([e.action for e in experiences if e is not None])
    rewards = np.vstack([e.reward for e in experiences if e is not None])
    next_states = np.vstack([e.next_state for e in experiences if e is not None])
    dones = np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)

    return (states, actions, rewards, next_states, dones)

  def __len__(self):
    """Return the current size of internal memory."""
    return len(self.memory)  
  
if __name__ == '__main__':
  
  act = KAgent(state_size=24, action_size=4)
  print("Mean of online weights {}".format([x.mean() for x in act.actor_online.get_weights()]))
  print("Mean of target weights {}".format([x.mean() for x in act.actor_target.get_weights()]))
  modw = [x * 1e-5 for x in act.actor_target.get_weights()]
  print("Modifying target weights")
  act.actor_target.set_weights(modw)
  print("Mean of target weights {}".format([x.mean() for x in act.actor_target.get_weights()]))
  tau = 0.5
  print("Soft update target with tau={}".format(tau))
  act.soft_copy_actor(tau=tau)
  print("Mean of target weights {}".format([x.mean() for x in act.actor_target.get_weights()]))
  