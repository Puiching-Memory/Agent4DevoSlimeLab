import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QTextEdit, QGroupBox, QTableWidget, QTableWidgetItem, QSizePolicy, QSplitter
)
from PyQt5.QtCore import Qt, QPoint, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QMouseEvent, QImage, QPixmap, QPainter, QPen, QColor

# 导入屏幕捕获模块
import mss

# 导入OpenAI库
from openai import OpenAI
import base64
import os
import json

# 导入pynput用于键盘和鼠标控制
from pynput import keyboard, mouse


class ScreenCaptureWidget(QWidget):
    """屏幕捕获展示窗口"""
    
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint)
        
        # 设置背景为半透明黑色，方便看到绘制的绿色框
        self.setStyleSheet("background-color: rgba(0, 0, 0, 30);")
        
    def capture_screen_region(self, x: int, y: int, width: int, height: int) -> np.ndarray:
        """捕获屏幕指定区域"""
        with mss.mss() as sct:
            # 注意：需要捕获窗口下方的内容，所以需要调整坐标
            # 由于窗口是置顶的，我们需要捕获窗口所在位置下方的内容
            monitor = {"top": y, "left": x, "width": width, "height": height}
            try:
                sct_img = sct.grab(monitor)
                img = np.array(sct_img)
                return img
            except Exception as e:
                print(f"屏幕捕获失败: {e}")
                return None
                
    def paintEvent(self, event):
        """绘制事件 - 只绘制边框"""
        painter = QPainter()
        painter.begin(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 绘制窗口可见范围的边框
        pen = QPen(QColor(0, 255, 0))  # 绿色边框
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawRect(0, 0, self.width() - 1, self.height() - 1)
        
        painter.end()
        
    def resizeEvent(self, event):
        """窗口大小改变事件"""
        super().resizeEvent(event)
        
    def closeEvent(self, event):
        """窗口关闭事件"""
        event.accept()


class TransparentWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("透明左侧区域示例")
        self.setGeometry(100, 100, 1000, 550)
        
        # 设置窗口属性以支持透明并保持置顶
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        
        # 添加窗口拖拽功能所需的变量
        self.drag_position = QPoint()
        self.resizing = False
        self.resize_direction = None
        self.margin = 10
        self.waiting_for_v = True
        
        # 自动迭代标志（移除自动迭代，改为手动按V键）
        self.auto_iterate = False
        
        # 初始化键盘和鼠标控制器
        self.keyboard_controller = keyboard.Controller()
        self.mouse_controller = mouse.Controller()
        
        # 存储当前按下的键
        self.pressed_keys = set()
        
        # 从外部文件加载游戏规则
        self.game_rules = self.load_game_rules()
        
        # 初始化对话历史
        self.messages = []
        # 添加初始系统消息（包含游戏规则）
        self.messages.append({
            "role": "user", 
            "content": [
                {"type": "text", "text": f"{self.game_rules}"}
            ]
        })
        
        # 初始化OpenAI客户端
        self.api_key = os.environ.get("DASHSCOPE_API_KEY", "")
        if not self.api_key:
            print("警告: 未设置DASHSCOPE_API_KEY环境变量")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        
        # 创建中央窗口部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建主布局（水平布局）
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # 创建左侧透明区域
        self.create_left_panel()
        
        # 创建右侧LLM控制台区域
        self.create_right_panel()
        
        # 添加左右两侧到主布局
        main_layout.addWidget(self.left_panel, 3)
        main_layout.addWidget(self.right_panel, 1)
        
        # 启动键盘监听器
        self.keyboard_listener = keyboard.Listener(on_press=self.on_key_press)
        self.keyboard_listener.start()
        
        print("系统已启动。按V键开始一帧操作，等待LLM决策并执行。")
    
    def load_game_rules(self):
        """从外部文件加载游戏规则"""
        try:
            with open("game_rules.txt", "r", encoding="utf-8") as f:
                return f.read().strip()
        except FileNotFoundError:
            return "未找到游戏规则文件"
        except Exception as e:
            return f"加载游戏规则时出错: {e}"
    
    def create_left_panel(self):
        """创建左侧透明面板"""
        self.left_panel = ScreenCaptureWidget()
        self.left_panel.setStyleSheet("""
            QWidget {
                background-color: rgba(0, 0, 0, 30);
                border: 2px solid #00ff00;
            }
        """)
    
    def create_right_panel(self):
        """创建右侧LLM控制台面板"""
        self.right_panel = QWidget()
        self.right_panel.setStyleSheet("""
            QWidget {
                background-color: white;
            }
        """)
        self.right_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        layout = QVBoxLayout(self.right_panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(5)
        
        title_label = QLabel("LLM 控制台")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; margin: 5px;")
        title_label.setFixedHeight(25)
        layout.addWidget(title_label)
        
        splitter = QSplitter(Qt.Vertical)
        splitter.setContentsMargins(0, 0, 0, 0)
        splitter.setHandleWidth(3)
        
        # 上下文历史区域
        context_widget = QWidget()
        context_layout = QVBoxLayout(context_widget)
        context_layout.setContentsMargins(0, 0, 0, 0)
        context_layout.setSpacing(3)
        
        context_label = QLabel("上下文历史:")
        context_label.setStyleSheet("font-weight: bold; margin-top: 0px; margin-bottom: 3px;")
        context_label.setFixedHeight(18)
        context_layout.addWidget(context_label)
        
        self.context_history_text = QTextEdit()
        self.context_history_text.setReadOnly(True)
        self.context_history_text.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.context_history_text.setFixedHeight(140)
        context_layout.addWidget(self.context_history_text)
        
        splitter.addWidget(context_widget)
        
        # 最后动作区域
        action_widget = QWidget()
        action_layout = QVBoxLayout(action_widget)
        action_layout.setContentsMargins(0, 0, 0, 0)
        action_layout.setSpacing(3)
        
        action_label = QLabel("最后动作:")
        action_label.setStyleSheet("font-weight: bold; margin-top: 0px; margin-bottom: 3px;")
        action_label.setFixedHeight(18)
        action_layout.addWidget(action_label)
        
        self.last_action_text = QTextEdit()
        self.last_action_text.setFixedHeight(50)
        self.last_action_text.setReadOnly(True)
        self.last_action_text.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.last_action_text.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.last_action_text.setLineWrapMode(QTextEdit.WidgetWidth)
        action_layout.addWidget(self.last_action_text)
        
        splitter.addWidget(action_widget)
        
        # 对话次数区域
        dialog_widget = QWidget()
        dialog_layout = QVBoxLayout(dialog_widget)
        dialog_layout.setContentsMargins(0, 0, 0, 0)
        dialog_layout.setSpacing(3)
        
        dialog_label = QLabel("对话次数:")
        dialog_label.setStyleSheet("font-weight: bold; margin-top: 0px; margin-bottom: 3px;")
        dialog_label.setFixedHeight(18)
        dialog_layout.addWidget(dialog_label)
        
        self.dialog_count_label = QLabel("0")
        self.dialog_count_label.setFixedHeight(25)
        dialog_layout.addWidget(self.dialog_count_label)
        
        splitter.addWidget(dialog_widget)
        
        # 系统信息区域
        info_widget = QWidget()
        info_layout = QVBoxLayout(info_widget)
        info_layout.setContentsMargins(0, 0, 0, 0)
        info_layout.setSpacing(3)
        
        info_group = QGroupBox("系统信息")
        info_group_layout = QVBoxLayout()
        info_group_layout.setSpacing(3)
        info_group.setLayout(info_group_layout)
        
        self.window_info = QLabel("窗口位置: (0, 0)\n窗口大小: 0 x 0")
        self.window_info.setFixedHeight(35)
        info_group_layout.addWidget(self.window_info)
        
        self.status_info = QLabel("状态: 等待V键按下")
        self.status_info.setFixedHeight(18)
        info_group_layout.addWidget(self.status_info)
        
        info_layout.addWidget(info_group)
        splitter.addWidget(info_widget)
        
        layout.addWidget(splitter)
        
        self.info_timer = QTimer(self)
        self.info_timer.timeout.connect(self.update_info_display)
        self.info_timer.start(200)
    
    
    def update_info_display(self):
        """更新右侧信息显示"""
        try:
            # 更新窗口信息
            x, y = self.x(), self.y()
            w, h = self.width(), self.height()
            self.window_info.setText(f"窗口位置: ({x}, {y})\n窗口大小: {w} x {h}")
        except Exception as e:
            pass
    
    def process_frame(self):
        """处理一帧操作：截图->LLM决策->执行动作->等待下次V键"""
        # 首先自动按下V键，确保游戏状态同步
        self.keyboard_controller.press('v')
        QApplication.processEvents()
        QTimer.singleShot(50, lambda: self.keyboard_controller.release('v'))
        
        try:
            self.send_to_llm_and_execute()
        except Exception as e:
            print(f"处理帧时出错: {e}")
        finally:
            # 释放所有按下的键
            for key in list(self.pressed_keys):
                self.keyboard_controller.release(key)
            self.pressed_keys.clear()
            
            # 重置状态，等待下一次V键按下
            self.waiting_for_v = True
            self.status_info.setText("状态: 等待V键按下")
            self.status_info.setStyleSheet("font-family: Consolas; color: blue;")
            
    def on_key_press(self, key):
        """键盘按键事件处理"""
        try:
            if key == keyboard.KeyCode.from_char('v') or key == keyboard.KeyCode.from_char('V'):
                if self.waiting_for_v:
                    # 按下V键后处理一帧
                    self.waiting_for_v = False
                    self.status_info.setText("状态: 处理中...")
                    self.status_info.setStyleSheet("font-family: Consolas; color: orange;")
                    # 强制更新显示
                    self.status_info.repaint()
                    # 在主线程中处理
                    QTimer.singleShot(0, self.process_frame)
        except AttributeError:
            pass
    
    def capture_screen(self):
        """捕获屏幕指定区域"""
        # 获取左侧面板的位置和大小
        x = self.left_panel.x() + self.x()
        y = self.left_panel.y() + self.y()
        w = self.left_panel.width()
        h = self.left_panel.height()
        
        # 确保捕获区域有效
        if w <= 0 or h <= 0:
            return None
        
        with mss.mss() as sct:
            # 注意：需要捕获窗口下方的内容，所以需要调整坐标
            # 由于窗口是置顶的，我们需要捕获窗口所在位置下方的内容
            monitor = {"top": y, "left": x, "width": w, "height": h}
            try:
                sct_img = sct.grab(monitor)
                img = np.array(sct_img)
                return img
            except Exception as e:
                print(f"屏幕捕获失败: {e}")
                return None
                
    def send_to_llm_and_execute(self):
        """发送屏幕截图到LLM进行分析并执行决策"""
        try:
            # 检查API密钥是否已设置
            if not self.api_key:
                print("API密钥未设置，跳过LLM请求")
                return
            
            # 在每次执行前捕获新的屏幕截图
            image = self.capture_screen()
            
            if image is None:
                print("屏幕截图失败，跳过LLM请求")
                return
            
            # 将图像转换为base64编码
            image_base64 = self.numpy_to_base64(image)
            
            # 构建消息内容，添加当前截图到消息历史
            self.messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                ]
            })
            
            # 限制消息历史长度，避免超出模型上下文限制
            # 保留系统消息（游戏规则）和最近几轮对话
            if len(self.messages) > 4:  # 1个系统消息 + n轮对话
                # 保留系统消息和最近n条消息
                self.messages = [self.messages[0]] + self.messages[-3:]

            # 发送请求到LLM
            completion = self.client.chat.completions.create(
                model="qwen3-vl-235b-a22b-instruct",
                messages=self.messages
            )
            
            # 获取响应
            response_text = completion.choices[0].message.content
            
            # 将LLM响应添加到消息历史
            self.messages.append({
                "role": "assistant",
                "content": response_text
            })
            
            # 更新对话次数
            current_count = int(self.dialog_count_label.text())
            new_count = current_count + 1
            self.dialog_count_label.setText(str(new_count))
            
            # 当对话次数达到20次时，重启游戏并清空上下文
            if new_count >= 20:
                self.restart_game_and_clear_context()
                return  # 重启后直接返回，不执行常规动作
            
            # 更新最后动作
            self.last_action_text.setPlainText(response_text)
            
            print(f"LLM响应: {response_text}")
            
            # 执行动作
            self.execute_action(response_text)
            
        except Exception as e:
            error_msg = f"发送到LLM时出错: {e}"
            print(error_msg)
            # 更新最后动作显示错误信息
            self.last_action_text.setPlainText(error_msg)
            pass
    
    def restart_game_and_clear_context(self):
        """重启游戏并清空上下文"""
        # 长按R键1秒来重启游戏
        self.keyboard_controller.press('r')
        QApplication.processEvents()
        QTimer.singleShot(1000, self._release_r_and_clear_context)
        
        # 更新界面显示
        self.last_action_text.setPlainText("对话次数达到20次，正在重启游戏...")
        self.status_info.setText("状态: 重启游戏中...")
        self.status_info.setStyleSheet("font-family: Consolas; color: red;")
    
    def _release_r_and_clear_context(self):
        """释放R键并清空上下文"""
        # 释放R键
        self.keyboard_controller.release('r')
        
        # 清空上下文历史
        self.context_history_text.setPlainText("")
        
        # 重置对话次数
        self.dialog_count_label.setText("0")
        
        # 清空消息历史，只保留系统消息（游戏规则）
        if self.messages:
            self.messages = [self.messages[0]]
        
        # 更新界面显示
        self.last_action_text.setPlainText("游戏已重启，上下文已清空")
        self.status_info.setText("状态: 等待V键按下")
        self.status_info.setStyleSheet("font-family: Consolas; color: blue;")
        
        print("游戏已重启，上下文已清空")
    
    def execute_action(self, action_text):
        """根据LLM的响应执行相应的动作"""
        action_result = ""
        try:
            # 将原始响应显示在"最后动作"区域
            self.last_action_text.setPlainText(action_text)
            
            # 尝试解析JSON格式的响应
            action_data = json.loads(action_text)
            thought = action_data.get("thought", "")
            actions = action_data.get("action", [])
            
            # 如果action不是列表，将其转换为列表
            if isinstance(actions, str):
                # 解析类似 "[A,D,Space,Mouse-R:[123,456],Mouse-L:[123,456]]" 的字符串
                actions_str = actions.strip()
                if actions_str.startswith('[') and actions_str.endswith(']'):
                    actions_str = actions_str[1:-1]
                actions = [action.strip() for action in actions_str.split(',') if action.strip()]
            
            if not isinstance(actions, list):
                actions = [actions]
                
            action_results = []
            for action in actions:
                result = self._execute_single_action(action)
                action_results.append(result)
            
            action_result = "; ".join(action_results)
            
        except json.JSONDecodeError:
            # 如果不是有效的JSON，尝试解析为普通文本
            # 将原始响应显示在"最后动作"区域
            self.last_action_text.setPlainText(action_text)
            action_result = self._parse_and_execute_text_actions(action_text)
        
        # 交换文本：将原本添加到上下文历史的内容放到最后动作文本框中
        self.last_action_text.setPlainText(action_result)
        
        # 交换文本：将原本的最后动作内容（即LLM响应）添加到上下文历史中
        current_history = self.context_history_text.toPlainText()
        if current_history:
            new_history = current_history + "\n" + action_text
        else:
            new_history = action_text
            
        # 限制上下文历史为最近5次对话
        history_lines = new_history.split("\n")
        if len(history_lines) > 5:
            # 保留最近的5条记录
            new_history = "\n".join(history_lines[-5:])
            
        self.context_history_text.setPlainText(new_history)
        
        # 自动滚动到底部
        scrollbar = self.context_history_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _execute_single_action(self, action):
        """执行单个动作"""
        action = action.strip()
        
        # 解析动作中的时长参数，例如 A:100 表示按下A键100毫秒
        duration = 50  # 默认持续时间50毫秒
        if ':' in action:
            parts = action.split(':')
            if len(parts) == 2:
                action_name = parts[0]
                try:
                    duration = int(parts[1])
                    action = action_name
                except ValueError:
                    pass  # 如果时长不是有效数字，则忽略
        else:
            action_name = action
        
        # 处理普通的键盘动作
        if action.upper() == "A":
            self.keyboard_controller.press('a')
            self.pressed_keys.add('a')
            # 对于需要持续按下的键，我们不自动释放，而是在帧结束时统一释放
            return f"执行动作: 向左移动 (A)"
        elif action.upper() == "D":
            self.keyboard_controller.press('d')
            self.pressed_keys.add('d')
            # 对于需要持续按下的键，我们不自动释放，而是在帧结束时统一释放
            return f"执行动作: 向右移动 (D)"
        elif action.upper() == "SPACE":
            self.keyboard_controller.press(keyboard.Key.space)
            self.pressed_keys.add(keyboard.Key.space)
            # 对于需要持续按下的键，我们不自动释放，而是在帧结束时统一释放
            return f"执行动作: 跳跃 (Space)"
        elif action.upper() == "W":
            self.keyboard_controller.press('w')
            self.pressed_keys.add('w')
            # 对于需要持续按下的键，我们不自动释放，而是在帧结束时统一释放
            return f"执行动作: 向上移动/跳出 (W)"
        elif action.upper() == "S":
            self.keyboard_controller.press('s')
            self.pressed_keys.add('s')
            # 对于需要持续按下的键，我们不自动释放，而是在帧结束时统一释放
            return f"执行动作: 向下移动/潜入 (S)"
        elif action.upper() == "R":
            self.keyboard_controller.press('r')
            QApplication.processEvents()
            QTimer.singleShot(1000, lambda: self.keyboard_controller.release('r'))  # 重启键固定1秒
            return "执行动作: 重启游戏 (R:1000ms)"
        elif action.upper() == "V":
            # 自动按下V键，用于触发下一帧
            self.keyboard_controller.press('v')
            QApplication.processEvents()
            QTimer.singleShot(50, lambda: self.keyboard_controller.release('v'))
            return "执行动作: 触发下一帧 (V:50ms)"
        
        # 处理鼠标动作 (不区分大小写)
        elif action.upper().startswith("MOUSE-R:[") and action.endswith("]"):
            coords_str = action[9:-1]  # 提取坐标部分
            # 支持逗号和分号分隔符
            coords = [c.strip() for c in coords_str.replace(';', ',').split(',') if c.strip()]
            if len(coords) == 2:
                try:
                    # 获取相对坐标
                    x = int(coords[0].strip())
                    y = int(coords[1].strip())
                    # 转换为绝对坐标（窗口位置+相对坐标）
                    abs_x = self.x() + x
                    abs_y = self.y() + y
                    # 记录当前是否正在按住A或D键
                    is_a_pressed = 'a' in self.pressed_keys
                    is_d_pressed = 'd' in self.pressed_keys
                    
                    # 如果正在按住A或D键，先释放
                    if is_a_pressed:
                        self.keyboard_controller.release('a')
                        self.pressed_keys.discard('a')
                    if is_d_pressed:
                        self.keyboard_controller.release('d')
                        self.pressed_keys.discard('d')
                        
                    # 移动鼠标并点击右键
                    self.mouse_controller.position = (abs_x, abs_y)
                    self.mouse_controller.click(mouse.Button.right)
                    
                    # 点击完成后重新按下之前按住的键
                    if is_a_pressed:
                        self.keyboard_controller.press('a')
                        self.pressed_keys.add('a')
                    if is_d_pressed:
                        self.keyboard_controller.press('d')
                        self.pressed_keys.add('d')
                        
                    return f"执行动作: 吸取平台 (鼠标右键 [{x},{y}] -> [{abs_x},{abs_y}])"
                except ValueError:
                    return f"未识别的鼠标坐标格式: {action}"
            else:
                return f"无效的鼠标坐标: {action}"
                
        elif action.upper().startswith("MOUSE-L:[") and action.endswith("]"):
            coords_str = action[9:-1]  # 提取坐标部分
            # 支持逗号和分号分隔符
            coords = [c.strip() for c in coords_str.replace(';', ',').split(',') if c.strip()]
            if len(coords) == 2:
                try:
                    # 获取相对坐标
                    x = int(coords[0].strip())
                    y = int(coords[1].strip())
                    # 转换为绝对坐标（窗口位置+相对坐标）
                    abs_x = self.x() + x
                    abs_y = self.y() + y
                    # 记录当前是否正在按住A或D键
                    is_a_pressed = 'a' in self.pressed_keys
                    is_d_pressed = 'd' in self.pressed_keys
                    
                    # 如果正在按住A或D键，先释放
                    if is_a_pressed:
                        self.keyboard_controller.release('a')
                        self.pressed_keys.discard('a')
                    if is_d_pressed:
                        self.keyboard_controller.release('d')
                        self.pressed_keys.discard('d')
                        
                    # 移动鼠标并点击左键
                    self.mouse_controller.position = (abs_x, abs_y)
                    self.mouse_controller.click(mouse.Button.left)
                    
                    # 点击完成后重新按下之前按住的键
                    if is_a_pressed:
                        self.keyboard_controller.press('a')
                        self.pressed_keys.add('a')
                    if is_d_pressed:
                        self.keyboard_controller.press('d')
                        self.pressed_keys.add('d')
                        
                    return f"执行动作: 喷射平台 (鼠标左键 [{x},{y}] -> [{abs_x},{abs_y}])"
                except ValueError:
                    return f"未识别的鼠标坐标格式: {action}"
            else:
                return f"无效的鼠标坐标: {action}"
        
        else:
            return f"未识别的动作: {action}"

    def _parse_and_execute_text_actions(self, action_text):
        """解析并执行文本形式的动作"""
        # 查找形如 [A,D,Space,Mouse-R:[123,456],Mouse-L:[123,456]] 的动作列表
        import re
        
        # 查找方括号内的内容
        pattern = r'\[([^\]]+)\]'
        match = re.search(pattern, action_text)
        
        if match:
            actions_str = match.group(1)
            actions = [action.strip() for action in actions_str.split(',') if action.strip()]
            
            action_results = []
            for action in actions:
                result = self._execute_single_action(action)
                action_results.append(result)
            
            return "; ".join(action_results)
        else:
            # 如果没有找到标准格式，尝试查找关键词
            action_results = []
            
            if "A" in action_text.upper() or "左" in action_text:
                self.keyboard_controller.press('a')
                self.pressed_keys.add('a')
                action_results.append("执行动作: 向左移动 (A)")
            if "D" in action_text.upper() or "右" in action_text:
                self.keyboard_controller.press('d')
                self.pressed_keys.add('d')
                action_results.append("执行动作: 向右移动 (D)")
            if "SPACE" in action_text.upper() or "空格" in action_text.upper() or "跳跃" in action_text:
                self.keyboard_controller.press(keyboard.Key.space)
                self.pressed_keys.add(keyboard.Key.space)
                action_results.append("执行动作: 跳跃 (Space)")
            if "W" in action_text.upper() or "向上" in action_text or "跳出" in action_text:
                self.keyboard_controller.press('w')
                self.pressed_keys.add('w')
                action_results.append("执行动作: 向上移动/跳出 (W)")
            if "S" in action_text.upper() or "向下" in action_text or "潜入" in action_text:
                self.keyboard_controller.press('s')
                self.pressed_keys.add('s')
                action_results.append("执行动作: 向下移动/潜入 (S)")
                
            # 尝试匹配鼠标动作（简单匹配）
            mouse_r_pattern = r'Mouse-R:\[(\d+),(\d+)\]'
            mouse_l_pattern = r'Mouse-L:\[(\d+),(\d+)\]'
            
            for match in re.finditer(mouse_r_pattern, action_text):
                x, y = match.groups()
                action_results.append(f"执行动作: 吸取平台 (鼠标右键 [{x},{y}])")
                
            for match in re.finditer(mouse_l_pattern, action_text):
                x, y = match.groups()
                action_results.append(f"执行动作: 喷射平台 (鼠标左键 [{x},{y}])")
            
            if action_results:
                return "; ".join(action_results)
            else:
                return f"未识别的动作: {action_text}"
    
    def numpy_to_base64(self, image: np.ndarray) -> str:
        """将numpy图像数组转换为base64编码"""
        _, buffer = cv2.imencode('.jpg', image)
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')
        return jpg_as_text
    
    def mousePressEvent(self, event: QMouseEvent):
        """鼠标按下事件"""
        if event.button() == Qt.LeftButton:
            self.drag_position = event.globalPos() - self.frameGeometry().topLeft()
            
            pos = event.pos()
            rect = self.rect()
            
            if pos.x() < self.margin:
                self.resize_direction = "left"
                self.resizing = True
            elif pos.x() > rect.width() - self.margin:
                self.resize_direction = "right"
                self.resizing = True
            elif pos.y() < self.margin:
                self.resize_direction = "top"
                self.resizing = True
            elif pos.y() > rect.height() - self.margin:
                self.resize_direction = "bottom"
                self.resizing = True
    
    def mouseMoveEvent(self, event: QMouseEvent):
        """鼠标移动事件"""
        if event.buttons() == Qt.LeftButton:
            if self.resizing and self.resize_direction:
                self._resize_window(event.globalPos())
            elif not self.resizing:
                self.move(event.globalPos() - self.drag_position)
    
    def mouseReleaseEvent(self, event: QMouseEvent):
        """鼠标释放事件"""
        self.resizing = False
        self.resize_direction = None
    
    def _resize_window(self, global_pos):
        """调整窗口大小"""
        if self.resize_direction == "left":
            if global_pos.x() < self.geometry().right() - 100:  # 最小宽度限制
                self.setGeometry(global_pos.x(), self.y(), 
                                self.geometry().right() - global_pos.x(), 
                                self.height())
        elif self.resize_direction == "right":
            width = max(global_pos.x() - self.x(), 100)  # 最小宽度限制
            self.resize(width, self.height())
        elif self.resize_direction == "top":
            if global_pos.y() < self.geometry().bottom() - 100:  # 最小高度限制
                self.setGeometry(self.x(), global_pos.y(),
                                self.width(), 
                                self.geometry().bottom() - global_pos.y())
        elif self.resize_direction == "bottom":
            height = max(global_pos.y() - self.y(), 100)  # 最小高度限制
            self.resize(self.width(), height)
            
    def closeEvent(self, event):
        """主窗口关闭事件"""
        if hasattr(self, 'keyboard_listener'):
            self.keyboard_listener.stop()
        
        if hasattr(self, 'left_panel'):
            self.left_panel.close()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = TransparentWindow()
    window.show()
    
    exit_code = app.exec_()
    sys.exit(exit_code)


if __name__ == '__main__':
    main()