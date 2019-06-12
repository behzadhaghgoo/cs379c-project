import random
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from .DQN import DQN, update_target
from .TDLoss import TDLoss
from .PrioritizedBuffer import PrioritizedBuffer
from .utils import *

beta_start = 0.4
beta_frames = 1000
BETA_BY_FRAME = lambda frame_idx: min(1.0, beta_start + frame_idx * (1.0 - beta_start) / beta_frames)

epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 500
EPSILON_BY_FRAME = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)


def test(val_env, noisyGame, eps, num_val_trials, current_model):
    rewards = []
    for i in range(num_val_trials):
        epsilon = 0
        episode_reward = 0
        state = val_env.reset()
        state = np.append(state, float(noisyGame))
        with torch.no_grad():
            while True:
                original_action = current_model.act(state, epsilon)

                if original_action != int(original_action):
                    original_action = original_action.numpy()[0]

                actual_action = original_action
                next_state, reward, done, _ = val_env.step(actual_action)
                next_state = np.append(next_state, float(noisyGame))

                if noisyGame:
                    reward += random.uniform(-1., 1.)

                state = next_state
                episode_reward += reward

                if done:
                    rewards.append(episode_reward)
                    break
    return np.mean(rewards)


def train(env, val_env, 
          method, var, mean,
          decision_eps,
          alpha, beta,
          hardcoded, cnn,
          invert_actions = False, num_frames = 10000, 
          num_val_trials = 10, batch_size = 32, gamma = 0.99, 
          num_trials = 10, USE_CUDA = False, device = "", eps = 1.):
    
    # TODO: comment out later
    num_trials = 1
    
    device = torch.device("cuda")

    """Args:"""
    losses = []
    all_rewards = []
    standard_val_rewards = []
    noisy_val_rewards = []
    states_count_ratios = []
    episode_reward = 0

    # Initialize state
    noisyGame = False
    state = env.reset()
    state = np.append(state, float(noisyGame)) # BACK IN
    meta_state = (state, float(noisyGame))

    # Initialize replay buffer, model, TD loss, and optimizers
    replay_buffer = PrioritizedBuffer(100000)
    # replay_buffer = PrioritizedBuffer(1000)
    if cnn:
        current_model = CnnDQN(env.observation_space.shape[0] + 1, env.action_space.n)
        target_model  = CnnDQN(env.observation_space.shape[0] + 1, env.action_space.n)
    else:
        current_model = DQN(env.observation_space.shape[0] + 1, env.action_space.n) # BACK IN
        target_model  = DQN(env.observation_space.shape[0] + 1, env.action_space.n) # BACK IN
    td_loss = TDLoss(method=method)
    optimizer = optim.Adam(current_model.parameters())
    
    # Multi GPU - Under Construction.
#     current_model = current_model.to(device)
#     target_model = target_model.to(device)

#     # Single GPU Code
    if USE_CUDA:
        current_model = nn.DataParallel(current_model, gpu_ids)
        target_model = nn.DataParallel(target_model, gpu_ids)
        current_model = current_model.cuda()
        target_model  = target_model.cuda()

    result_df = pd.DataFrame()
    theta = 1.
    power = theta
    all_standard_val_rewards = []
    all_proportions = []
    std_weights = []
    noisy_weights = []
    std_buffer_example_count = []
    noisy_buffer_example_count = []
    
    # TODO: restructure code to incorporate epsilon as experiment hyperparam
    # eps=.5

    for t in tqdm(range(num_trials)):
        print("trial number: {}".format(t))
        #print("is noisy game?", noisyGame)
        for frame_idx in range(1, num_frames + 1):
            epsilon = EPSILON_BY_FRAME(frame_idx)
            original_action = current_model.act(state, epsilon)

            # If in noisy environment, make action random with probability eps
            if noisyGame and random.uniform(0,1) < decision_eps:
                if invert_actions:
                    actual_action = 1 - original_action # invert
                else:
                    actual_action = original_action
            else:
                actual_action = original_action

            next_state, reward, done, _ = env.step(actual_action)

            # If in noisy environment, make reward completely random
            if noisyGame:
                reward *= np.random.normal(mean, var)

            next_state = np.append(next_state, float(noisyGame))
            meta_next_state = (next_state, float(noisyGame))
            replay_buffer.push(meta_state, original_action, reward, meta_next_state, done)

            meta_state = meta_next_state
            episode_reward += reward

            if done:
                if not hardcoded:
                    noisyGame = 1-noisyGame
                state = env.reset()
                state = np.append(state, float(noisyGame))
                meta_state = (state, float(noisyGame))
                all_rewards.append(episode_reward)
                episode_reward = 0

            if len(replay_buffer) > batch_size:
                beta = BETA_BY_FRAME(frame_idx)
                # TODO: to compare with PER paper, shouldn't this be every 4 frames?
                loss = td_loss.compute_td_loss(current_model, target_model, beta, replay_buffer, optimizer)
                losses.append(loss.data.tolist())

            if frame_idx % 200 == 0:
                all_standard_val_rewards.append(test(val_env, False, eps, num_val_trials, current_model))
                all_proportions.append(float(replay_buffer.states_count[1]) / (float(replay_buffer.states_count[1])  + float(replay_buffer.states_count[0])))
                weight_dict = replay_buffer.get_average_weight_by_env()
                std_weights.append(weight_dict['std_avg'])
                noisy_weights.append(weight_dict['noisy_avg'])
                std_buffer_example_count.append(weight_dict['std_count'])
                noisy_buffer_example_count.append(weight_dict['noisy_count'])
                
                # Verified that standard experience replay weights noisy examples more heavily and oversamples
                # TODO: run complete experiments and check improvement with proposed replay technique
#                 print(weight_dict)
#                 print('Noisy to std weight', noisy_weights[-1]/std_weights[-1])
#                 print('Proportion of noisy selected (long run): ', all_proportions[-1])
                
            #         plot(frame_idx, all_rewards, losses, standard_val_rewards, noisy_val_rewards, states_count_ratios)

            if frame_idx % 1000 == 0:
                update_target(current_model, target_model)
                print("frame_idx", frame_idx)

    result_df['trial_num'] = list(range(int(num_frames / 200))) * num_trials # 50 = 10000 frames / 200 frams
    result_df['frame'] = result_df['trial_num'] * 200
    result_df['val_reward'] = all_standard_val_rewards
    result_df['proportion'] = all_proportions
    result_df['std_weights'] = std_weights
    result_df['noisy_weights'] = noisy_weights
    result_df['std_buffer_example_count'] = std_buffer_example_count
    result_df['noisy_buffer_example_count'] = noisy_buffer_example_count
    return result_df
