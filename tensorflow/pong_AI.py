import numpy as np
import config
import pickle
import gym
import tensorflow as tf

env = gym.make('CartPole-v0')

now_img = np.zeros(config.pong_shape)
last_img = np.zeros(config.pong_shape)

num_frames_this_game = 0
num_episodes = 10


H = 200
D = 80*80
model = {}
model['W1'] = np.random.randn(H,D) / np.sqrt(D) # "Xavier" initialization
model['W2'] = np.random.randn(H) / np.sqrt(H)


grad_buffer = { k : np.zeros_like(v) for k,v in model.items() } # update buffers that add up gradients over a batch
rmsprop_cache = { k : np.zeros_like(v) for k,v in model.items() } # rmsprop memory
pass


