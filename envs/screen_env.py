import gymnasium as gym
import numpy as np
import mss
import pywinctl as pwc
from pynput.keyboard import Key, Controller as KeyboardController
from pynput.mouse import Button, Controller as MouseController
import time
import cv2
from ultralytics import YOLO

def capture_window(window_title: str) -> np.ndarray:
    windows = pwc.getWindowsWithTitle(window_title)
    if not windows:
        print(f'Window "{window_title}" not found!')
        return None

    win = windows[0]
    left, top, right, bottom = win.left, win.top, win.right, win.bottom
    width = right - left
    height = bottom - top
    # print(f'Capturing window "{window_title}" at ({left}, {top}), size ({width}x{height})')
    
    with mss.mss() as sct:
        monitor = {"top": top, "left": left, "width": width, "height": height}
        sct_img = sct.grab(monitor)
        img = np.array(sct_img)
        # mss.tools.to_png(sct_img.rgb, sct_img.size, output='capture.png')

    return img


class ScreenEnv(gym.Env):
    def __init__(
        self,
        window_name="DevoSlime Lab b0.4.64",
        render_mode=None
    ):
        self.window_name = window_name
        self.render_mode = render_mode
        self.screen_width = 790
        self.screen_height = 568

        # 获取窗口位置信息
        self._update_window_position()

        # 初始化键盘和鼠标控制器
        self.keyboard = KeyboardController()
        self.mouse = MouseController()
        
        # 初始化按键状态跟踪
        self.key_pressed = {}
        
        # 初始化YOLO模型
        self.model = YOLO("runs/slime_yolo/loop3/weights/best.pt")
        # 读取标签名称
        self.class_names = self._load_class_names()

        self.action_space = gym.spaces.Discrete(7 + 4 + 2) # 动作空间: AD + Space + SW + QE + Mouse(4+2)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.screen_height, self.screen_width, 3), dtype=np.uint8
        )
        
        # 动作映射表
        self.action_names = {
            0: "do_nothing",
            1: "press_a",
            2: "press_d",
            3: "press_space",
            4: "press_s",
            5: "press_w",
            6: "press_q",
            7: "press_e",
            8: "mouse_move_left",
            9: "mouse_move_right",
            10: "mouse_move_up",
            11: "mouse_move_down",
            12: "mouse_click_left",
            13: "mouse_click_right",
        }
        
        self.actions = {
            0: self._do_nothing,
            1: lambda: self._press_key('a'),
            2: lambda: self._press_key('d'),
            3: lambda: self._press_key(Key.space),
            4: lambda: self._press_key('s'),
            5: lambda: self._press_key('w'),
            6: lambda: self._press_key('q'),
            7: lambda: self._press_key('e'),
            8: lambda: self._move_mouse(-10, 0),
            9: lambda: self._move_mouse(10, 0),
            10: lambda: self._move_mouse(0, -10),
            11: lambda: self._move_mouse(0, 10),
            12: lambda: self._click_mouse(Button.left),
            13: lambda: self._click_mouse(Button.right),
        }
        
        # 时间相关变量
        self.step_count = 0
        self.total_reward = 0  # 累计总奖励
        self.max_steps = 50  # 最大步数，达到后重启游戏

    def _load_class_names(self):
        """加载类别名称"""
        # 从data.yaml中读取类别名称
        class_names = {}
        try:
            with open("data.yaml", "r", encoding="utf-8") as f:
                lines = f.readlines()
                reading_names = False
                for line in lines:
                    if line.strip() == "names:":
                        reading_names = True
                        continue
                    if reading_names:
                        if line.strip().startswith("#") or not line.strip():
                            continue
                        if ":" in line:
                            parts = line.strip().split(":")
                            if len(parts) == 2:
                                idx = int(parts[0].strip())
                                name = parts[1].strip()
                                class_names[idx] = name
                        else:
                            break
        except Exception as e:
            print(f"加载类别名称时出错: {e}")
            # 使用默认类别名称
            class_names = {
                0: "slime",
                1: "sign up"
                # 可以根据需要添加更多类别
            }
        return class_names

    def _update_window_position(self):
        """更新窗口位置信息"""
        windows = pwc.getWindowsWithTitle(self.window_name)
        if not windows:
            print(f'Window "{self.window_name}" not found!')
            # 使用默认值
            self.window_left = 0
            self.window_top = 0
            self.window_right = self.screen_width
            self.window_bottom = self.screen_height
        else:
            win = windows[0]
            self.window_left, self.window_top, self.window_right, self.window_bottom = win.left, win.top, win.right, win.bottom

    def step(self, action):
        # 执行动作
        print(f"Step {self.step_count}: Executing action - {self.action_names.get(action, 'unknown_action')} ({action})")
        self._execute_action(action)
        
        # 增加步骤计数
        self.step_count += 1
        
        # 获取新的观测值
        observation = self._get_observation()
        
        # 基于YOLO检测结果的距离奖励
        reward = self._calculate_detection_reward(observation)
        
        # 打印当前得分信息
        detection_reward = reward  # 为了保持打印语句的一致性
        print(f"Step {self.step_count}: "
              f"Detection reward: {detection_reward:.3f}, "
              f"Step reward: {reward:.3f}")
        
        # 检查是否达到最大步数
        terminated = False
        truncated = self.step_count >= self.max_steps
        
        # 如果达到最大步数，需要重启游戏
        if truncated:
            print(f"Reached maximum steps ({self.max_steps}), restarting game...")
        
        info = {}
        
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # 释放所有按下的键
        for key in list(self.key_pressed.keys()):
            if self.key_pressed[key]:
                self.keyboard.release(key)
                self.key_pressed[key] = False
                
        # 长按R键重启游戏
        self.keyboard.press('r')
        time.sleep(0.5)  # 持续按住R键
        self.keyboard.release('r')
        time.sleep(0.5)  # 等待游戏重启完成
        # 更新窗口位置（窗口可能移动）
        self._update_window_position()
        # 获取初始观测值
        observation = self._get_observation()
        info = {}

        # 重置计数器
        self.step_count = 0
        print("Environment reset - starting new episode")
        
        return observation, info
    
    def _calculate_detection_reward(self, image):
        """
        根据YOLO检测结果计算奖励
        距离"sign up"标签中心越近，得分越高
        """
        try:
            # 使用YOLO模型进行预测
            results = self.model.predict(source=image, verbose=False)
            
            # 解析检测结果
            sign_up_boxes = []
            player_box = None
            
            # 遍历所有检测结果
            for r in results:
                boxes = r.boxes
                if boxes is not None:
                    for i, box in enumerate(boxes):
                        class_id = int(box.cls[0])  # 类别ID
                        class_name = self.class_names.get(class_id, f"class_{class_id}")
                        
                        # 如果是"sign up"标签
                        if class_name == "sign up":
                            # 获取边界框坐标
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            sign_up_boxes.append((x1, y1, x2, y2))
                            
                        # 如果是玩家角色（slime）
                        elif class_name == "slime":
                            # 获取边界框坐标
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            player_box = (x1, y1, x2, y2)
            
            # 如果检测到了玩家和至少一个"sign up"标签
            if player_box is not None and len(sign_up_boxes) > 0:
                # 计算玩家中心点
                player_center_x = (player_box[0] + player_box[2]) / 2
                player_center_y = (player_box[1] + player_box[3]) / 2
                
                # 计算最近的"sign up"标签中心点
                min_distance = float('inf')
                for box in sign_up_boxes:
                    sign_center_x = (box[0] + box[2]) / 2
                    sign_center_y = (box[1] + box[3]) / 2
                    
                    # 计算欧几里得距离
                    distance = np.sqrt((player_center_x - sign_center_x)**2 + (player_center_y - sign_center_y)**2)
                    
                    if distance < min_distance:
                        min_distance = distance
                
                # 将距离转换为奖励（距离越近奖励越高）
                # 使用指数衰减函数，使接近目标时奖励增加更明显
                max_reward = 1.0
                # 当距离小于阈值时给予额外奖励，鼓励到达目标点
                if min_distance < 20:  # 像素距离小于20时认为已到达目标
                    distance_reward = max_reward * 2  # 额外奖励
                    print(f"Reached target! Distance: {min_distance:.1f}")
                else:
                    # 使用指数衰减函数，使接近目标时奖励增加更明显
                    distance_reward = max_reward * np.exp(-min_distance/200)
                
                print(f"Detection: Player at ({player_center_x:.1f}, {player_center_y:.1f}), "
                      f"Nearest 'sign up' at distance {min_distance:.1f}, "
                      f"Distance reward: {distance_reward:.3f}")
                
                return distance_reward
            
            # 如果没有检测到必要的对象，则无额外奖励
            print("Detection: No player or 'sign up' signs detected")
            return 0.0
            
        except Exception as e:
            print(f"计算检测奖励时出错: {e}")
            return 0.0
    
    def render(self):
        # 渲染环境，这里直接返回当前观测值
        return self._get_observation()
    
    def _get_observation(self):
        # 截取窗口图像作为观测值
        img = capture_window(self.window_name)
        img = cv2.resize(img, (self.screen_width, self.screen_height))
        
        # 如果图像是4通道的RGBA格式，转换为3通道BGR格式
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        return img
    
    def _execute_action(self, action):
        if action in self.actions:
            self.actions[action]()
        # 添加小延迟以便动作生效
        time.sleep(0.05)
    
    def _do_nothing(self):
        # 什么都不做
        pass
    
    def _press_key(self, key):
        # 持续按下并释放按键，而不是瞬时按下
        if key in self.key_pressed and self.key_pressed[key]:
            # 如果键已经被按下，先释放它
            self.keyboard.release(key)
            self.key_pressed[key] = False
        else:
            # 按下按键并保持一小段时间
            self.keyboard.press(key)
            self.key_pressed[key] = True
            time.sleep(0.1)  # 按下保持时间
            self.keyboard.release(key)
            self.key_pressed[key] = False
    
    def _move_mouse(self, dx, dy):
        # 移动鼠标
        current_x, current_y = self.mouse.position
        new_x = current_x + dx
        new_y = current_y + dy
        
        # 限制鼠标位置在窗口范围内
        new_x = max(self.window_left, min(self.window_right - 1, new_x))
        new_y = max(self.window_top, min(self.window_bottom - 1, new_y))
        
        self.mouse.position = (new_x, new_y)
    
    def _click_mouse(self, button):
        # 点击鼠标
        self.mouse.click(button, 1)