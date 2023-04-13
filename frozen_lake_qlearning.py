import numpy as np
import pandas as pd
import gym
import pygame
from algorithms.rl import RL
from algorithms.planner import Planner
from examples.test_env import TestEnv
from examples.grid_search import GridSearch
from examples.plots import Plots
import itertools
import time

problem_name = 'FrozenLake-v1'
grid_size = 8
algorithm = 'Q-Learning'
epsilon = 0.4
n_episodes = 10000
simulations=10
print('\n%s-%dx%d - Algorithm: %s' % (problem_name, grid_size, grid_size, algorithm))
print("running -- with epsilon decay: ", epsilon, " episodes: ", n_episodes)
# Define MDP problem
map_name = str(grid_size) + "x" + str(grid_size)
mdp_problem_params = {"map_name": map_name, "is_slippery": True}
mdp_problem = gym.make(problem_name, **mdp_problem_params)
start = time.time()
# Train Q-learner
Q, V, pi, Q_track, pi_track = RL(mdp_problem.env).q_learning(epsilon_decay_ratio=epsilon, n_episodes=n_episodes)
end = time.time()



def get_reward_simulation(mdp_problem, pi, simulations=50):
    test_scores = TestEnv.test_env(env=mdp_problem.env, render=False, user_input=False, pi=pi, n_iters=simulations)
    return sum(test_scores)

# Get reward over simulations
total_reward = get_reward_simulation(mdp_problem, pi, simulations)
performance_score = total_reward / simulations
# Get convergence plot
get_convergence_plot_qlearn(Q_track, problem_name, grid_size, algorithm)
# Get state values
Plots.grid_values_heat_map(V, "State Values")
df = get_policy_visual(pi, grid_size)
print(df)
print('Total rewards over %d simulations: %d (%.0f%%)' % (simulations, total_reward, performance_score*100))
run_time = end - start
result = {}
result['grid_size'] = grid_size
result['epsilon'] = epsilon
result['n_episodes'] = n_episodes
result['total_reward'] = total_reward
result['simulations'] = simulations
result['performance_score'] = performance_score
result['run_time'] = run_time
return Q, V, pi, Q_track, pi_track, result


# algorithm = 'Q-learning'
# PROBLEM_NAME = 'FrozenLake-v1'
# GRID_SIZES = [4, 8]
# epsilon_decay = [.4, .9]
# iters = [50000]
#
# problem_name = PROBLEM_NAME
# grid_size = 8
# print('\n%s-%dx%d - Algorithm: %s' % (problem_name, grid_size, grid_size, algorithm))
# # Define MDP problem
# map_name = str(grid_size) + "x" + str(grid_size)
# mdp_problem_params = {"map_name": map_name, "is_slippery": True}
# mdp_problem = gym.make(problem_name, **mdp_problem_params)
#
#
#
#
# def get_policy_visual(pi, grid_size):
#     action_dict = {0: 'left', 1: 'down', 2: 'right', 3: 'up'}
#     action_digit_list = [[] for _ in range(grid_size)]
#     action_desc_list = [[] for _ in range(grid_size)]
#     for x in range(grid_size):
#         for y in range(grid_size):
#             i = x + y * 4
#             print(x, y, i)
#             action_digit_list[y].append(pi(i))
#             action_desc_list[y].append(action_dict[pi(i)])
#     return pd.DataFrame(action_desc_list)
#
#
# # def Q_learning_grid_search(env, epsilon_decay, iters):
# for i in itertools.product(epsilon_decay, iters):
#     epsilon = i[0]
#     iter = i[1]
#     print("running -- with epsilon decay: ", epsilon, " iterations: ", iter)
#     Q, V, pi, Q_track, pi_track = RL(mdp_problem.env).q_learning(epsilon_decay_ratio=epsilon, n_episodes=iter)
#     get_policy_visual(pi, grid_size)
#     Plots.grid_values_heat_map(V, "State Values")