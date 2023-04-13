import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gym
import pygame
from algorithms.rl import RL
from algorithms.planner import Planner
from examples.test_env import TestEnv
from examples.grid_search import GridSearch
from examples.blackjack import Blackjack
from examples.plots import Plots
import seaborn as sns
import itertools
import time
import random
import warnings
warnings.filterwarnings("ignore")


def rl_train(mdp_problem, algorithm='Value Iteration', params={'n_iters': 1000}):
    if algorithm == 'Value Iteration':
        random.seed(200)
        V, V_track, pi = Planner(mdp_problem.P).value_iteration(**params)
        return V, V_track, pi
    elif algorithm == 'Policy Iteration':
        random.seed(200)
        V, V_track, pi = Planner(mdp_problem.P).policy_iteration(**params)
        return V, V_track, pi
    return None


def get_reward_simulation(mdp_problem, pi, simulations=50):
    test_scores = TestEnv.test_env(env=mdp_problem.env, render=False, user_input=False, pi=pi, n_iters=simulations,
                                   convert_state_obs=mdp_problem.convert_state_obs)
    total_reward = np.sum(test_scores)
    performance_score = total_reward / simulations
    #     TestEnv.test_env(env=blackjack.env, render=False, pi=pi, user_input=False,
    #                                    convert_state_obs=blackjack.convert_state_obs, n_iters=simulations)
    return test_scores, total_reward, performance_score


def iterate_rl_train(mdp_problem, algorithm, train_iterations=1000, simulations=50, params=None):
    results = pd.DataFrame(columns=['iteration', 'simulations', 'total_reward', 'performance_score'])
    for n in range(1, train_iterations):
        result = {}
        result['iteration'] = n
        result['simulations'] = simulations
        if params == None:
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


def get_convergence_plot(V_track, V, problem_name, algorithm):
    V_track_avg = []
    trained_iterations = V_track.shape[0]
    for i in range(V_track.shape[0]):
        V_track_avg.append(np.average(V_track[i]))
    for i in range(V.shape[0]):
        comparison = V_track[i] == V
        equal_arrays = comparison.all()
        if equal_arrays:
            break
    print('Converged after %d episodes' % i)
    plt.plot(range(1, i + 1, 1), V_track_avg[:i])
    plt.title(
        '%s - Convergence of V(s) using %s' % (problem_name, algorithm))
    plt.xlabel('episodes')
    plt.ylabel('V(s)')
    plt.show()


def get_policy_visual(pi):
    player_hands = []
    for i in range(4, 22, 1):
        player_hands.append('H' + str(i))
    for i in range(12, 22, 1):
        player_hands.append('S' + str(i))
    player_hands.append('BJ')
    action_dict = {0: 'stick', 1: 'hit'}
    action_digit_list = [[] for _ in range(29)]
    action_desc_list = [[] for _ in range(29)]

    for x in range(10):
        for y in range(29):
            i = x + y * 10
            action_digit_list[y].append(pi(i))
            action_desc_list[y].append(action_dict[pi(i)])
    df = pd.DataFrame(action_desc_list)
    df.columns = [2, 3, 4, 5, 6, 7, 8, 9, 'T', 'A']
    df.index = player_hands
    return df


def grid_values_heat_map(V, label):
    player_hands = []
    for i in range(4, 22, 1):
        player_hands.append('H' + str(i))
    for i in range(12, 22, 1):
        player_hands.append('S' + str(i))
    player_hands.append('BJ')

    x = 29
    y = 10
    data = np.around(np.array(V).reshape((x, y)), 2)
    df = pd.DataFrame(data=data)
    df.columns = [2, 3, 4, 5, 6, 7, 8, 9, 'T', 'A']
    df.index = player_hands
    sns.heatmap(df, annot=True, annot_kws={'fontsize': 6}).set_title(label)
    plt.show()


def get_result_rl_algo(problem_name, algorithm='Value Iteration', train_iterations=1000, simulations=50):
    print('\n%s - Algorithm: %s' % (problem_name, algorithm))
    # Define MDP problem
    blackjack = Blackjack()

    # # Not sure if I'm going to use  this section...
    # result = iterate_rl_train(mdp_problem,
    #                           algorithm,
    #                           train_iterations=train_iterations,
    #                           simulations=simulations)
    # plt.plot(result['iteration'], result['total_reward'])

    # Train algorithm
    params = {'n_iters': train_iterations}
    V, V_track, pi = rl_train(blackjack, algorithm, params)
    # Get reward over simulations
    test_scores, total_reward, performance_score = get_reward_simulation(blackjack, pi, simulations)
    # Get convergence plot
    get_convergence_plot(V_track, V, problem_name, algorithm)
    # Get state values
    grid_values_heat_map(V, "State Values")
    df = get_policy_visual(pi)
    print(df)
    print('Total rewards over %d simulations: %d (%.0f%%)' % (simulations, total_reward, performance_score * 100))
    print(test_scores)
    return V, V_track, pi, total_reward, performance_score, test_scores


def get_convergence_plot_qlearn(Q_track, problem_name, algorithm):
    Q_track_avg = []
    trained_iterations = Q_track.shape[0]
    for i in range(Q_track.shape[0]):
        Q_track_avg.append(np.average(Q_track[i]))
    plt.plot(range(trained_iterations), Q_track_avg[:trained_iterations])
    plt.title(
        '%s - Convergence of Q(s) using %s' % (problem_name, algorithm))
    plt.xlabel('episodes')
    plt.ylabel('Q(s)')
    plt.show()


def get_result_q_learning_algo(problem_name, epsilon, n_episode, simulations, algorithm='Q-Learning'):
    print('\n%s - Algorithm: %s' % (problem_name, algorithm))
    print("running -- with epsilon decay: ", epsilon, " episodes: ", n_episode)
    # Define MDP problem
    blackjack = Blackjack()

    start = time.time()
    # Train Q-learner
    np.random.seed(200)
    Q, V, pi, Q_track, pi_track = RL(blackjack.env).q_learning(blackjack.n_states, blackjack.n_actions,
                                                               blackjack.convert_state_obs,
                                                               epsilon_decay_ratio=epsilon, n_episodes=n_episode)
    end = time.time()
    if np.sum(V) > 0:
        # Get reward over simulations
        test_scores, total_reward, performance_score = get_reward_simulation(blackjack, pi, simulations)
    else:
        test_scores, total_reward, performance_score = [], 0, 0
    #         test_scores = []
    #         total_reward = 0
    #         performance_score = 0
    # Get convergence plot
    get_convergence_plot_qlearn(Q_track, problem_name, algorithm)
    # Get state values
    grid_values_heat_map(V, "State Values")
    df = get_policy_visual(pi)
    print(df)
    print('Total rewards over %d simulations: %d (%.0f%%)' % (simulations, total_reward, performance_score * 100))
    run_time = end - start
    result = {}
    result['epsilon'] = epsilon
    result['n_episodes'] = n_episode
    result['total_reward'] = total_reward
    result['simulations'] = simulations
    result['performance_score'] = performance_score
    result['run_time'] = run_time
    return Q, V, pi, Q_track, pi_track, result


def grid_search_q_learn(problem_name, epsilons, n_episodes, simulations, algorithm='Q-Learning'):
    result_df = pd.DataFrame(
        columns=['epsilon', 'n_episodes', 'total_reward', 'simulations', 'performance_score', 'run_time'])
    for grid in itertools.product(epsilons, n_episodes):
        epsilon = grid[0]
        n_episode = grid[1]
        _, _, _, _, _, result = get_result_q_learning_algo(problem_name,
                                                           epsilon,
                                                           n_episode,
                                                           simulations,
                                                           algorithm='Q-Learning')
        result_df = result_df._append(result, ignore_index=True)
    return result_df

PROBLEM_NAME = 'Blackjack-v1'
SIMULATIONS = 200
EPSILON_DECAY = [.4]
# ITERS = [500, 1000, 2000, 3000, 4000, 5000, 10000, 30000, 60000, 90000,
#         500, 1000, 2000, 3000, 4000, 5000, 10000, 30000, 60000, 90000,
#         500, 1000, 2000, 3000, 4000, 5000, 10000, 30000, 60000, 90000,
#         500, 1000, 2000, 3000, 4000, 5000, 10000, 30000, 60000, 90000,
#         500, 1000, 2000, 3000, 4000, 5000, 10000, 30000, 60000, 90000]
ITERS = [30000, 30000, 30000, 30000, 30000]
result_df = grid_search_q_learn(problem_name=PROBLEM_NAME,
                                epsilons = EPSILON_DECAY,
                                n_episodes = ITERS,
                                simulations = SIMULATIONS,
                                algorithm = 'Q-Learning')