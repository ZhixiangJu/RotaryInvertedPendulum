#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dqn_rotary_ip_real_v4.py

这是一个“现实世界”倒立摆DQN训练脚本，用来平替原先的 dqn_rotary_ip_sim_v4。
不同点：
 - 环境由 gym.make(...) 换成 真实物理环境 RotaryInvertedPendulumMotorDiscreteV2Env
 - 其余逻辑（神经网络的选择、经验回放、保存训练曲线与模型等）保持不变

运行方式：与原先相同，把此脚本放在和所有依赖相同的文件夹下，执行：
    python dqn_rotary_ip_real_v4.py

鞠志翔
2025.3.7
"""

import os
import time
import pickle
import random
import datetime
import numpy as np
import tensorflow as tf
from collections import deque

# 1. 导入你写的“真实物理”倒立摆环境(同名替换原gym版本)
#    确保文件 RotaryInvertedPendulumMotorDiscreteV2Env.py 中的类也叫 RotaryInvertedPendulumMotorDiscreteV2Env
from RotaryInvertedPendulumMotorDiscreteV2Env import RotaryInvertedPendulumMotorDiscreteV2Env

# 2. 其它工具和脚本，与原来基本一致
from neural_network_structure.show_choose_nn import ShowChooseNN
from save_data_class.save_plot_dqn_info import SavePlotDQNInfo


# --------------------------------------------------------------------
# 一些与训练脚本相关的辅助函数(写pickle/读pickle/保存训练信息等)
# --------------------------------------------------------------------
def write_data_pickle(data, filename):
    with open(filename, 'wb') as f_obj:
        pickle.dump(data, f_obj)

def read_data_pickle(filename):
    with open(filename, 'rb') as f_obj:
        data = pickle.load(f_obj)
    return data

def get_current_py_name():
    """获取当前脚本文件名(不带后缀)"""
    full_name = os.path.basename(__file__)
    only_name = full_name.split(".")[0]
    return only_name

def save_training_info_txt(filename, env_name, nn_name):
    """保存训练环境、神经网络、脚本等信息到txt文件"""
    curr_time = datetime.datetime.now()
    time_str = datetime.datetime.strftime(curr_time, '%Y_%m_%d_%H_%M_%S')
    py_name = get_current_py_name()
    with open(filename, 'w') as f_obj:
        f_obj.write(time_str + '\n')
        f_obj.write("env_name=" + str(env_name) + '\n')
        f_obj.write("nn_name=" + str(nn_name) + '\n')
        f_obj.write("python script name: " + py_name)


# --------------------------------------------------------------------
# 核心：DQN训练函数，平替原 dqn_rotary_ip_sim_v4 里的 main loop
# --------------------------------------------------------------------
def DQNagent(model, target_md, parameters_dict):
    """
    训练逻辑与原先相同，只是 env 不再用 gym.make(...)，
    而是用 RealInvertedPendulumMotorDiscreteV2Env(...)。
    """
    # 解包训练参数
    num_episodes = parameters_dict['num_episodes']
    max_len_episode = parameters_dict['max_len_episode']
    batch_size = parameters_dict['batch_size']
    learning_rate = parameters_dict['learning_rate']
    gamma = parameters_dict['gamma']
    initial_epsilon = parameters_dict['initial_epsilon']
    final_epsilon = parameters_dict['final_epsilon']
    decay_rate = parameters_dict['decay_rate']
    max_len_replay_buffer = parameters_dict['max_len_replay_buffer']
    actions_num = parameters_dict['actions_num']
    nn_name = parameters_dict['neural_network_name']
    # env_version_name = parameters_dict['env_version_name']  # 在真实环境时可不再使用

    # 创建保存文件夹
    base_file_path = os.path.dirname(os.path.abspath(__file__))
    curr_time = datetime.datetime.now()
    time_str = datetime.datetime.strftime(curr_time, '%Y_%m_%d_%H_%M_%S')
    store_file_path = os.path.join(base_file_path, 'save_model', 'dqn_real_' + time_str)
    os.makedirs(store_file_path, exist_ok=True)

    # 准备数据记录对象
    dqn_data_saver = SavePlotDQNInfo(store_file_path, parameters_dict)

    # 初始化真实物理环境
    #   (如果串口不在 /dev/ttyUSB0，就改成自己的端口；波特率改成与你硬件一致)
    env = RotaryInvertedPendulumMotorDiscreteV2Env(
        port="/dev/ttyUSB0",
        baudrate=115200
    )

    # 优化器、经验回放
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    replay_buffer = deque(maxlen=max_len_replay_buffer)
    model_list = []  # 用于记录那些成功保持平衡较久的 episode 索引(可选)

    # 训练循环
    for episode_id in range(num_episodes):
        state = env.reset()
        # 按指数衰减来计算epsilon
        epsilon = max(
            final_epsilon + (initial_epsilon - final_epsilon) * np.exp(-decay_rate * episode_id),
            final_epsilon
        )
        total_reward = 0

        for t in range(max_len_episode):
            # epsilon-greedy
            if random.random() < epsilon:
                action = env.action_space.sample()  # 随机
            else:
                q_values = model.predict(np.expand_dims(state, axis=0))
                action = np.argmax(q_values[0])

            # 与真实环境交互
            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            # 存储到 replay buffer
            replay_buffer.append((state, action, reward, next_state, 1 if done else 0))
            state = next_state

            if done:
                # 如果摆杆倒下或到达极限，就结束这个回合
                break

            # 从经验池采样并更新网络
            if len(replay_buffer) >= batch_size:
                batch_data = random.sample(replay_buffer, batch_size)
                batch_state, batch_action, batch_reward, batch_next_state, batch_done = map(np.array, zip(*batch_data))

                # 目标 Q 值
                q_next = target_md(batch_next_state)
                y = batch_reward + gamma * tf.reduce_max(q_next, axis=1) * (1 - batch_done)

                with tf.GradientTape() as tape:
                    q_current = model(batch_state)
                    # 选取对应动作的 Q(s,a)
                    q_action = tf.reduce_sum(q_current * tf.one_hot(batch_action, depth=actions_num), axis=1)
                    loss = tf.keras.losses.mean_squared_error(y, q_action)
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # 如果本回合走到 max_len_episode 还没 done，可以认为是不错的表现
        if t == max_len_episode - 1:
            model_list.append(episode_id)

        dqn_data_saver.add_data(total_reward, t)
        print(f"Episode {episode_id}, Epsilon={epsilon:.4f}, Steps={t}, Reward={total_reward:.2f}")

        # 每回合更新目标网络
        target_md.set_weights(model.get_weights())

    # 训练结束，保存训练曲线和模型等信息
    dqn_data_saver.plot_dqn_info_rewards()
    dqn_data_saver.plot_dqn_info_steps()
    dqn_data_saver.save_dqn_info()

    model.save_weights(os.path.join(store_file_path, 'model_final', 'my_weights'), save_format='tf')
    write_data_pickle(model_list, os.path.join(store_file_path, 'good_model_index_pickle.txt'))
    save_training_info_txt(os.path.join(store_file_path, 'training_info.txt'), "RealRotaryInvertedPendulum", nn_name)

    # 关闭环境
    env.close()


# --------------------------------------------------------------------
# 主入口，与原 dqn_rotary_ip_sim_v4.py 完全对应
# --------------------------------------------------------------------
if __name__ == '__main__':
    print("================== DQN Training (Real World) ==================")
    show_choose_nn = ShowChooseNN()
    fail, model, nn_structure_name = show_choose_nn.choose_nn()
    if fail == 1:
        print("Exit.")
    else:
        # 同步 target 网络
        ret_val, target_md = show_choose_nn.choose_nn_no_ipt(nn_structure_name)
        if ret_val == 1:
            print("Error creating target network. Exit.")
            exit()
        # 先跑一次空的forward，初始化网络
        _ = model.predict(np.expand_dims([0.0, 0.0, 0.0, 0.0], axis=0))
        _ = target_md.predict(np.expand_dims([0.0, 0.0, 0.0, 0.0], axis=0))
        target_md.set_weights(model.get_weights())

        # 准备训练参数（可以根据需求修改）
        parameters_dict = {
            'num_episodes':            3000,
            'max_len_episode':         2000,
            'batch_size':              256,
            'learning_rate':           0.001,
            'gamma':                   0.90,
            'initial_epsilon':         1.0,
            'final_epsilon':           0.001,
            'decay_rate':              0.002,
            'max_len_replay_buffer':   10000,
            'actions_num':             model.actions_num,
            'neural_network_name':     model.nn_name,
            # 'env_version_name':      "RotaryInvertedPendulumMotorDiscreteV2Env" # 原仿真环境用，现实环境可不必
        }

        print("Parameters:")
        for k, v in parameters_dict.items():
            print(f"  {k}: {v}")

        print("\nStart real-world training...\n")
        DQNagent(model, target_md, parameters_dict)