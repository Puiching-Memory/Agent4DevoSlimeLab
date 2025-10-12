import gymnasium as gym
import numpy as np
from stable_baselines3 import A2C,PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import sys
import os

# 添加envs目录到Python路径，以便导入ScreenEnv
sys.path.append(os.path.join(os.path.dirname(__file__), 'envs'))
from screen_env import ScreenEnv

def train_screen_env():
    print("初始化训练环境...")
    
    # 创建训练环境
    # 注意：由于这是一个基于屏幕捕获的环境，我们不使用render_mode进行训练以提高速度和稳定性
    train_env = DummyVecEnv([lambda: ScreenEnv(window_name="DevoSlime Lab b0.4.64")])
    
    # 由于这是一个基于图像的环境，我们使用CnnPolicy
    model = PPO(
        "CnnPolicy",
        train_env,
        verbose=1,
    )
    
    print("开始训练...")
    # 开始训练
    model.learn(
        total_timesteps=50000,
        log_interval=1
    )
    
    # 保存最终模型
    model.save("a2c_screen_env")
    print("模型已保存为 a2c_screen_env")


if __name__ == "__main__":
    train_screen_env()
