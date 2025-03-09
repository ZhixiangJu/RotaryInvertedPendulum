#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""？？》？？？〉
deploy_trained_model.py

演示如何在真实摆上运行训练好的模型(推力不再随机探索)：
  - 加载已有的 model
  - 进入一个循环：用 model 的贪心策略选择动作 -> env.step(action)
  - 如同一个长回合(或多个episode), 直到手动退出

作者: 你自己
时间: 2023/xx/xx
"""

import os
import time
import numpy as np
import tensorflow as tf

from RotaryInvertedPendulumMotorDiscreteV2Env import RotaryInvertedPendulumMotorDiscreteV2Env
from neural_network_structure.show_choose_nn import ShowChooseNN


def run_deploy(model_weight_path, env, max_steps=5000):
    """
    加载model权重后，用贪心策略在真实倒立摆上运行。
    :param model_weight_path: 训练好的model参数文件，例如 "save_model/dqn_real_.../model_final/my_weights"
    :param env: 实际物理环境
    :param max_steps: 最多步数
    """
    # 1) 加载权重
    model = None
    show_choose_nn = ShowChooseNN()
    fail, model, nn_structure_name = show_choose_nn.choose_nn()  # 选择网络结构
    if fail == 1:
        print("Error: no valid model chosen.")
        return
    # 构建网络
    dummy_in = np.zeros((1,4), dtype=np.float32)
    model(dummy_in)  # forward一次完成初始化

    # 加载训练好的权重
    model.load_weights(model_weight_path)
    print(f"Loaded model weights from {model_weight_path}")

    # 2) 开始测试
    print("Start running the trained policy... Press Ctrl+C to stop.")
    obs = env.reset()
    step_count = 0

    while True:
        # 贪心选择动作
        q_values = model.predict(np.expand_dims(obs, axis=0))
        action = np.argmax(q_values[0])

        # 与环境交互
        obs, reward, done, _ = env.step(action)
        step_count += 1

        # 如果 done，则可以复位(或者你也可以选择退出循环)
        if done:
            print("Pendulum fell down or out-of-bounds. Reset environment.")
            obs = env.reset()

        # 或者加个上限，避免无限运行
        if step_count >= max_steps:
            print(f"Reached {max_steps} steps, stop now.")
            break

        time.sleep(0.01)  # 控制实时节奏

    env.close()

if __name__ == "__main__":
    # 你可以修改这里的路径为实际训练好的路径
    # 例如 "save_model/dqn_real_2023_XX_XX_XX_XX_XX/model_final/my_weights"
    model_weight_path = input("Input the path to your trained model weights: ")

    # 创建真实摆环境
    env = RotaryInvertedPendulumMotorDiscreteV2Env(port="/dev/ttyUSB0", baudrate=115200)

    run_deploy(model_weight_path, env, max_steps=5000)