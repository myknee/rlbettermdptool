import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gym
import pygame
from algorithms.rl import RL
from algorithms.planner import Planner
from examples.test_env import TestEnv
from examples.grid_search import GridSearch
from examples.plots import Plots
import itertools
import time
import warnings
warnings.filterwarnings("ignore")


def rl_train(mdp_problem, algorithm='Value Iteration', params={'n_iters': 1000}):
    if algorithm == 'Value Iteration':
        V, V_track, pi = Planner(mdp_problem.env.P).value_iteration(**params)
        return V, V_track, pi
    elif algorithm == 'Policy Iteration':
        V, V_track, pi = Planner(mdp_problem.env.P).policy_iteration(**params)
        return V, V_track, pi
    return None


def get_reward_simulation(mdp_problem, pi, simulations=50):
    test_scores = TestEnv.test_env(env=mdp_problem.env, render=False, user_input=False, pi=pi, n_iters=simulations)
    return sum(test_scores)


def iterate_rl_train(mdp_problem, algorithm, train_iterations=1000, simulations=50, params=None):
    results = pd.DataFrame(columns=['iteration', 'simulations', 'total_reward', 'performance_score'])
    for n in range(1, train_iterations):
        result = {}
        result['iteration'] = n
        result['simulations'] = simulations
        if params==None:
            params = {'n_iters': n}
        else:
            params['n_iters'] = n
        # Train algorithm
        V, V_track, pi = rl_train(mdp_problem, algorithm, params)
        # Get reward over simulations
        if sum(V) == 0:  # Case when model has not converged yet
            result['total_reward'] = 0
            result['performance_score'] = 0
        else:
            result['total_reward'] = get_reward_simulation(mdp_problem, pi)
            result['performance_score'] = result['total_reward'] / simulations
        results = results._append(result, ignore_index=True)
    return results


def get_convergence_plot(V_track, problem_name, algorithm, grid_size):
    V_track_max = []
    prev_max_V_track = 0
    for i in range(V_track.shape[0]):
        max_V_track = max(V_track[i])
        if max_V_track >= prev_max_V_track:
            V_track_max.append(max_V_track)
        if max_V_track < prev_max_V_track:
            trained_iterations = i
            break
    plt.plot(range(V_track.shape[trained_iterations]), V_track_max[:trained_iterations])
    plt.title(
        '%s-%dx%d - Convergence of V(s) using %s' % (problem_name, grid_size, grid_size, algorithm))
    plt.xlabel('iterations')
    plt.ylabel('V(s)')
    plt.show()


def get_policy_visual(pi, grid_size):
    action_dict = {0: 'left', 1: 'down', 2: 'right', 3: 'up'}
    action_digit_list = [[] for _ in range(grid_size)]
    action_desc_list = [[] for _ in range(grid_size)]
    for x in range(grid_size):
        for y in range(grid_size):
            i = x + y * 4
            print(x, y, i)
            action_digit_list[y].append(pi(i))
            action_desc_list[y].append(action_dict[pi(i)])
    return pd.DataFrame(action_desc_list)


def get_result_rl_algo(problem_name, grid_size, algorithm='Value Iteration', train_iterations=1000, simulations=50):
    # Define MDP problem
    map_name = str(grid_size) + "x" + str(grid_size)
    mdp_problem_params = {"map_name": map_name, "is_slippery": True}
    mdp_problem = gym.make(problem_name, **mdp_problem_params)

    # # Not sure if I'm going to use  this section...
    # result = iterate_rl_train(mdp_problem,
    #                           algorithm,
    #                           train_iterations=train_iterations,
    #                           simulations=simulations)
    # plt.plot(result['iteration'], result['total_reward'])

    # Train algorithm
    params = {'n_iters': train_iterations}
    V, V_track, pi = rl_train(mdp_problem, algorithm, params)
    # Get reward over simulations
    total_reward = get_reward_simulation(mdp_problem, pi)
    performance_score = total_reward / simulations
    # Get convergence plot
    get_convergence_plot(V_track, problem_name, algorithm, grid_size)
    # Get state values
    Plots.grid_values_heat_map(V, "State Values")
    df = get_policy_visual(pi, grid_size)
    print(df)
    return V, V_track, pi, total_reward, performance_score


# Declare variables unique to MDP problem
PROBLEM_NAME = 'FrozenLake-v1'
GRID_SIZES = [4, 8]
ALGORITHMS = ['Value Iteration', 'Policy Iteration']
TRAIN_ITERATIONS = 1000
SIMULATIONS = 50

for grid_size in GRID_SIZES:
    for algorithm in ALGORITHMS:
        get_result_rl_algo(problem_name=PROBLEM_NAME, grid_size=grid_size, algorithm=algorithm)

map_name = str(grid_size) + "x" + str(grid_size)
mdp_problem_params = {"map_name": map_name, "is_slippery": True}
mdp_problem = gym.make(PROBLEM_NAME, **mdp_problem_params)
V, V_track, pi = rl_train(mdp_problem, algorithm, {'n_iters':1000})

V_track_max = []
prev_max_V_track = 0
for i in range(V_track.shape[0]):
    max_V_track = max(V_track[i])
    if max_V_track >= prev_max_V_track:
        V_track_max.append(max_V_track)
        prev_max_V_track = max_V_track
    if max_V_track < prev_max_V_track:
        trained_iterations = i
        break
plt.plot(range(trained_iterations), V_track_max[:trained_iterations])
plt.title(
    '%s-%dx%d - Convergence of V(s) using %s' % (problem_name, grid_size, grid_size, algorithm))
plt.xlabel('iterations')
plt.ylabel('V(s)')
plt.show()