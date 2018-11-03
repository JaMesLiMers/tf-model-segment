#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import numpy as np
import pandas as pd
import time


N_STATES = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]  # 9种states
ACTIONS = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]  # 可能的actions
EPSILON = 0.9  # greedy policy 的概率
ALPHA = 0.1  # for learning rate 的大小
GAMMA = 0.9  # for discount factor 的大小
episode = 0


def build_q_table(n_states, actions):
    """
    To build a q-table for initialization
    :param n_states: for all states actor may have
    :param actions:  for all action's value
    :return:  a q-table
    """
    table = pd.DataFrame(
        np.zeros((len(n_states), len(actions))),
        columns=actions,index=n_states
    )
    return table

# Initialize Q(s,a) arbitrary

def choose_action(state, q_table, EPSILON):
    """
    To choose an action for current state
    :param state: current state
    :param q_table: input q-table
    :return: a choice of action
    """
    state_actions = q_table.loc[state, :] # currently state's table
    if (np.random.uniform() > EPSILON) or (state_actions.all() == 0): # use greedy policy
        action_name = np.random.choice(ACTIONS)
    else:
        action_name = state_actions.argmax()
    return action_name # 为什么是一个字符串呢？？？？？？？？？？ 学习一下panda的构造

def get_env_feedback(A, Fps_past, Fps_now):
    R = Fps_now - Fps_past
    S_ = int(A)
    return S_, R # return reward and next state

def update_env(episode, step_counter):
    # This is how environment be updated
    interaction = "\n\nEpisode %s: total_steps = %s\n\n" % (episode+1, step_counter) # stop episode
    print('\r{}'.format(interaction))

def rl():
    q_table = build_q_table(N_STATES, ACTIONS) # Initialize Q(s, a)
    for episode in range(MAX_EPSILON): # Repeat (for each episode):
        step_counter = 0 # init step
        S = 0 # init state
        is_terminated = False # is stop?
        update_env(S, episode, step_counter) # update env
        while not is_terminated: # repeat (for each step in episode)
            A = choose_action(S, q_table) # choose a from s using greedy policy
            S_, R = get_env_feedback(S, A) # take action, observe R, S_
            q_predict = q_table.loc[S, A]  # observe R, S（保存s时间q_table的值）
            if _s != "terminal":
                q_target = R + GAMMA * q_table.iloc[S_, :].max() # 已得到的 - 未得到的reward乘衰减值
            else:
                q_target = R     #  实际的(状态-行为)值 (回合结束)
                is_terminated = True    # terminate this episode

            q_table.loc[S, A] += ALPHA * (q_target - q_predict)  # q_table 更新
            S = S_  # 探索者移动到下一个 state

            update_env(S, episode, step_counter + 1)  # 环境更新

            step_counter += 1
    return q_table



if __name__ == "__main__":
    q_table = rl()
    print('\r\nQ-table:\n')

