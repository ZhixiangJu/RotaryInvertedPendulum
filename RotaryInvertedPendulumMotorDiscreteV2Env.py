#juzhixiang/3/5
import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import time
import serial

class RotaryInvertedPendulumMotorDiscreteV2Env(gym.Env):
    """
    与原来的 RotaryInvertedPendulumMotorDiscreteV2Env 同名的“真实物理”环境：
      - step() 不用本地动力学，而是发送PWM到硬件，并从传感器读取下一时刻状态
      - reward函数、done条件、pwm_list、render() 等基本逻辑都与原代码相同
      - 这样在训练脚本中几乎无需改动，只要把 env = gym.make(...) 换成
        env = RotaryInvertedPendulumMotorDiscreteV2Env(port="...", baudrate=...) 即可

    注意：
      1) 下位机(Arduino/STM32等)需要实现对应的串口协议：
         - "RESET": 复位摆杆
         - "SET_ACTION,<pwm>": 设定电机PWM
         - "GET_STATE": 返回 "theta,theta_dot,alpha,alpha_dot"
      2) 这里的摆杆长度、角度阈值仅用于判定 done 与渲染，可根据硬件情况修改
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self,
                 port="/dev/ttyUSB0",
                 baudrate=115200,
                 pwm_list=None):
        """
        初始化物理倒立摆环境：
          :param port: 串口端口，如 '/dev/ttyUSB0'
          :param baudrate: 波特率，如 115200
          :param pwm_list: 若不传，则默认和原先一致 [-255, -204, ..., 255]
        """
        # ===== 1. 串口连接硬件 =====
        self.port = port
        self.baudrate = baudrate
        self.ser = serial.Serial(port=self.port, baudrate=self.baudrate, timeout=1)
        time.sleep(2)  # 等待串口稳定

        # ===== 2. 原环境的核心参数（仅用于done判定、渲染等）=====
        self.length = 0.0675         # 用于渲染杆的长度(半长)
        self.pwm_threshold = 255     # PWM不应超过此绝对值
        self.tau = 0.005             # step间隔 (硬件执行+读取传感器的周期)
        self.kinematics_integrator = 'euler'

        # 角度阈值：alpha ±30度(≈0.52 rad)，theta ±2π(≈6.28)
        self.alpha_threshold_radians = 30 * 2 * math.pi / 360
        self.length_1 = 0.15189
        self.theta_threshold = 2 * math.pi
        self.theta_dot_threshold = 20

        if pwm_list is None:
            # 与原版本一致(10档离散PWM)
            self.pwm_list = [-255, -204, -153, -102, -51, 51, 102, 153, 204, 255]
        else:
            self.pwm_list = pwm_list

        # 动作空间(Discrete)
        self.action_space = spaces.Discrete(len(self.pwm_list))

        # 观测空间：4维 [theta, theta_dot, alpha, alpha_dot]
        high = np.array([
            self.theta_threshold * 2,
            np.finfo(np.float32).max,
            self.alpha_threshold_radians * 2,
            np.finfo(np.float32).max
        ], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # 一些初始化
        self.seed()
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None

    def seed(self, seed=None):
        """Gym风格的随机种子函数。"""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        """
        执行动作 -> 发送PWM到硬件 -> 读取真实状态 -> 计算 reward、done -> 返回
        """
        assert self.action_space.contains(action), f"{action} invalid"

        # 1) 将离散索引转为PWM值
        pwm = self.pwm_list[action]

        # 2) 通过串口向下位机发送PWM
        self._apply_action_to_hardware(pwm)

        # 3) 给硬件一点时间执行(或根据需要在下位机处理好)
        time.sleep(self.tau)

        # 4) 从硬件读取 [theta, theta_dot, alpha, alpha_dot]
        theta, theta_dot, alpha, alpha_dot = self._get_state_from_hardware()
        self.state = (theta, theta_dot, alpha, alpha_dot)

        # 5) 判定 done
        done = bool(
            theta < -self.theta_threshold
            or theta > self.theta_threshold
            or alpha < -self.alpha_threshold_radians
            or alpha > self.alpha_threshold_radians
            or abs(theta_dot) > self.theta_dot_threshold
        )

        # 6) 计算奖励
        if not done:
            # 未倒
            reward = self._get_reward_all_states(theta, theta_dot, alpha, alpha_dot, 0, action)
        elif self.steps_beyond_done is None:
            # 刚倒下(第一帧)
            self.steps_beyond_done = 0
            reward = self._get_reward_all_states(theta, theta_dot, alpha, alpha_dot, 1, action)
        else:
            # 后续帧
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this environment has returned done=True."
                    "You should always call 'reset()' once you receive 'done=True' -- any further steps are undefined."
                )
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state, dtype=np.float32), reward, done, {}

    def reset(self):
        """
        复位：向硬件发送 RESET 命令，让摆杆回到初始位置(或手动配合)；然后读取一次传感器。
        """
        self.steps_beyond_done = None
        self.ser.write(b"RESET\n")
        time.sleep(1.0)  # 视硬件情况调整

        # 从硬件读取初始状态(若需要额外随机，也可在此加逻辑)
        self.state = self._get_state_from_hardware()
        return np.array(self.state, dtype=np.float32)

    def render(self, mode='human'):
        """
        仅用于可视化（如果需要）。与原版render逻辑相同：简单画一个转杆动画。
        """
        screen_width = 600
        screen_height = 400

        world_width = self.length_1 * 2 * math.pi
        scale = screen_width / world_width
        carty = 100
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            l, r, t, b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)

            l, r, t, b = -polewidth/2, polewidth/2, polelen - polewidth/2, -polewidth/2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.poletrans = rendering.Transform(translation=(0, cartheight/4.0))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)

            axle = rendering.make_circle(polewidth/2)
            axle.add_attr(self.poletrans)
            axle.add_attr(self.carttrans)
            axle.set_color(.5, .5, .8)
            self.viewer.add_geom(axle)

            track = rendering.Line((0, carty), (screen_width, carty))
            track.set_color(0, 0, 0)
            self.viewer.add_geom(track)
            self._pole_geom = pole

        if self.state is None:
            return None

        # 更新贴图
        pole = self._pole_geom
        l, r, t, b = -polewidth/2, polewidth/2, polelen - polewidth/2, -polewidth/2
        pole.v = [(l, b), (l, t), (r, t), (r, b)]

        theta, theta_dot, alpha, alpha_dot = self.state

        cartx = self.length_1 * theta * scale + screen_width / 2.0
        # 如果想让小车跑到屏幕另一侧，你也可以用 while cartx>screen_width: cartx-=screen_width
        self.carttrans.set_translation(cartx, carty)
        # 摆杆绕水平轴转动 alpha
        self.poletrans.set_rotation(-alpha)

        return self.viewer.render(return_rgb_array=(mode=='rgb_array'))

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
        # 关闭串口
        self.ser.close()

    # ============== 原先的奖励函数，完全照搬 ==============
    def _get_reward_all_states(self, theta, theta_dot, alpha, alpha_dot, isdone, action_index):
        """
        计算奖励：
        与原有代码保持一致：k_1 theta^2 + k_2 theta_dot^2 + ...
        """
        k_1 = 0.00125
        k_2 = 0.000125
        k_3 = 8.2070
        k_4 = 0.002
        const = 1.3
        c_1 = 0.0
        temp_1 = k_1 * theta * theta
        temp_2 = k_2 * theta_dot * theta_dot
        temp_3 = k_3 * alpha * alpha
        temp_4 = k_4 * alpha_dot * alpha_dot
        reward = const - (temp_1 + temp_2 + temp_3 + temp_4) + c_1 * isdone
        return reward

    # ============== 发送PWM/读取硬件状态的辅助函数 ==============
    def _apply_action_to_hardware(self, pwm):
        """
        向下位机发送 "SET_ACTION,<pwm>\n" 命令
        下位机收到后应驱动电机进行正负方向转动
        """
        cmd = f"SET_ACTION,{pwm}\n"
        self.ser.write(cmd.encode())

    def _get_state_from_hardware(self):
        """
        向下位机发送 "GET_STATE" 命令，返回 "theta,theta_dot,alpha,alpha_dot\n"
        并解析float，若失败返回(0,0,0,0)
        """
        self.ser.write(b"GET_STATE\n")
        line = self.ser.readline().decode().strip()  # 例如: "0.12,0.01,-0.08,0.0"
        try:
            parts = line.split(',')
            if len(parts) == 4:
                theta_val = float(parts[0])
                theta_dot_val = float(parts[1])
                alpha_val = float(parts[2])
                alpha_dot_val = float(parts[3])
            else:
                theta_val, theta_dot_val, alpha_val, alpha_dot_val = (0.0, 0.0, 0.0, 0.0)
        except:
            theta_val, theta_dot_val, alpha_val, alpha_dot_val = (0.0, 0.0, 0.0, 0.0)

        return (theta_val, theta_dot_val, alpha_val, alpha_dot_val)