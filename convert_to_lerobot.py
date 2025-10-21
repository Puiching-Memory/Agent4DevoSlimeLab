"""
MCAP to LeRobotDataset Converter

将 MCAP 格式数据转换为 LeRobotDataset 格式。
支持以下模态：
- 屏幕捕获（视频帧）
- 鼠标事件（位置、点击）
- 键盘事件（按键）
"""

import json
import os
import logging
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import cv2

from mcap.reader import make_reader

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ObservationFrame:
    """单个观察帧数据"""
    timestamp_ns: int
    frame_idx: int
    screen_image_path: Optional[str] = None
    mouse_x: Optional[int] = None
    mouse_y: Optional[int] = None
    mouse_pressed: Optional[bool] = None
    keyboard_keys: Dict[str, bool] = field(default_factory=dict)
    raw_data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'timestamp_ns': self.timestamp_ns,
            'frame_idx': self.frame_idx,
            'screen_image_path': self.screen_image_path,
            'mouse': {
                'x': self.mouse_x,
                'y': self.mouse_y,
                'pressed': self.mouse_pressed
            },
            'keyboard': self.keyboard_keys,
            'raw_data': self.raw_data
        }


@dataclass
class ActionFrame:
    """单个动作帧数据"""
    timestamp_ns: int
    action_idx: int
    mouse_dx: Optional[float] = None
    mouse_dy: Optional[float] = None
    mouse_click: Optional[str] = None  # 'left', 'right', 'scroll', None
    keyboard_pressed: List[str] = field(default_factory=list)
    keyboard_released: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'timestamp_ns': self.timestamp_ns,
            'action_idx': self.action_idx,
            'mouse': {
                'dx': self.mouse_dx,
                'dy': self.mouse_dy,
                'click': self.mouse_click
            },
            'keyboard': {
                'pressed': self.keyboard_pressed,
                'released': self.keyboard_released
            }
        }


class MCAPToLeRobotConverter:
    """MCAP 到 LeRobotDataset 转换器"""
    
    def __init__(
        self,
        mcap_path: str,
        output_dir: str,
        dataset_name: str = "desktop_agent",
        video_fps: int = 30,
        max_messages: Optional[int] = None
    ):
        """
        初始化转换器
        
        Args:
            mcap_path: MCAP 文件路径
            output_dir: 输出目录
            dataset_name: 数据集名称
            video_fps: 视频帧率
            max_messages: 最大消息数（None 表示全部）
        """
        self.mcap_path = Path(mcap_path)
        self.output_dir = Path(output_dir)
        self.dataset_name = dataset_name
        self.video_fps = video_fps
        self.max_messages = max_messages
        
        # 确保输出目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir = self.output_dir / "images"
        self.images_dir.mkdir(exist_ok=True)
        
        # 数据收集
        self.observations: Dict[int, ObservationFrame] = {}
        self.actions: Dict[int, ActionFrame] = {}
        self.channels_map: Dict[int, Any] = {}
        self.schemas_map: Dict[int, Any] = {}
        self.messages_by_topic: Dict[str, List[Dict]] = defaultdict(list)
        
        # 跟踪时间戳
        self.frame_counter = 0
        self.action_counter = 0
        self.first_timestamp: Optional[int] = None
        self.last_timestamp: Optional[int] = None
        
        # 维持状态
        self.last_mouse_pos: Tuple[int, int] = (0, 0)
        self.keyboard_state: Dict[str, bool] = {}
        self.screen_frame_cache: Dict[int, str] = {}  # 时间戳到图像路径的映射
    
    def parse_mcap(self) -> Tuple[Dict, Dict, Dict]:
        """
        解析 MCAP 文件并收集所有数据
        
        Returns:
            (channels_map, schemas_map, messages_by_topic)
        """
        logger.info(f"开始解析 MCAP 文件: {self.mcap_path}")
        
        with open(self.mcap_path, "rb") as f:
            reader = make_reader(f)
            
            message_count = 0
            
            for item in reader.iter_messages():
                if self.max_messages and message_count >= self.max_messages:
                    logger.info(f"达到最大消息数限制: {self.max_messages}")
                    break
                
                if len(item) == 3:
                    schema, channel, message = item
                    
                    # 记录通道和模式
                    if channel.id not in self.channels_map:
                        self.channels_map[channel.id] = channel
                        logger.info(f"发现新通道 - ID: {channel.id}, Topic: {channel.topic}")
                    
                    if schema.id not in self.schemas_map:
                        self.schemas_map[schema.id] = schema
                        logger.debug(f"发现新模式 - ID: {schema.id}, Name: {schema.name}")
                    
                    # 转换时间戳
                    timestamp_ns = message.log_time
                    if self.first_timestamp is None:
                        self.first_timestamp = timestamp_ns
                    self.last_timestamp = timestamp_ns
                    
                    # 处理数据
                    self._process_message(channel, schema, message, timestamp_ns)
                    
                    message_count += 1
                    if message_count % 100 == 0:
                        logger.debug(f"已处理 {message_count} 条消息")
        
        logger.info(f"MCAP 文件解析完成")
        logger.info(f"总消息数: {message_count}")
        logger.info(f"通道数: {len(self.channels_map)}")
        logger.info(f"模式数: {len(self.schemas_map)}")
        logger.info(f"各 Topic 消息数: {dict(self.messages_by_topic)}")
        
        return self.channels_map, self.schemas_map, dict(self.messages_by_topic)
    
    def _process_message(self, channel, schema, message, timestamp_ns: int):
        """处理单个消息"""
        topic = channel.topic
        
        try:
            if channel.message_encoding == 'json':
                data = json.loads(message.data.decode('utf-8'))
            else:
                data = message.data
            
            # 根据 Topic 类型处理消息
            if 'screen' in topic.lower() or 'capture' in topic.lower():
                self._process_screen_message(data, timestamp_ns, topic)
            elif 'mouse' in topic.lower():
                self._process_mouse_message(data, timestamp_ns, topic)
            elif 'keyboard' in topic.lower():
                self._process_keyboard_message(data, timestamp_ns, topic)
            else:
                logger.debug(f"未知 Topic 类型: {topic}")
            
            self.messages_by_topic[topic].append({
                'timestamp_ns': timestamp_ns,
                'data': data,
                'schema': schema.name if schema else 'unknown'
            })
            
        except Exception as e:
            logger.warning(f"处理消息时出错 (Topic: {topic}): {e}")
    
    def _process_screen_message(self, data: Dict, timestamp_ns: int, topic: str):
        """处理屏幕捕获消息"""
        try:
            # 创建观察帧（如果还不存在）
            if timestamp_ns not in [obs.timestamp_ns for obs in self.observations.values()]:
                frame = ObservationFrame(
                    timestamp_ns=timestamp_ns,
                    frame_idx=self.frame_counter,
                    raw_data={'screen': data, 'topic': topic}
                )
                self.observations[timestamp_ns] = frame
                self.frame_counter += 1
            else:
                # 更新现有帧的屏幕数据
                for obs in self.observations.values():
                    if obs.timestamp_ns == timestamp_ns:
                        obs.raw_data['screen'] = data
                        obs.raw_data['screen_topic'] = topic
                        break
            
            # 处理不同的图像源
            if 'image_path' in data:
                self._save_screen_image(timestamp_ns, data['image_path'])
            elif 'image_base64' in data:
                self._process_base64_image(timestamp_ns, data['image_base64'])
            elif 'media_ref' in data:
                # 处理 OWA 格式的 media_ref（指向视频文件）
                media_ref = data['media_ref']
                if isinstance(media_ref, dict) and 'uri' in media_ref:
                    video_path = media_ref['uri']
                    pts_ns = media_ref.get('pts_ns', timestamp_ns)
                    self._extract_video_frame_from_media_ref(timestamp_ns, video_path, pts_ns)
                
        except Exception as e:
            logger.debug(f"处理屏幕消息时出错: {e}")
    
    def _process_mouse_message(self, data: Dict, timestamp_ns: int, topic: str):
        """处理鼠标消息"""
        try:
            # 创建或更新观察帧
            if timestamp_ns not in [obs.timestamp_ns for obs in self.observations.values()]:
                frame = ObservationFrame(
                    timestamp_ns=timestamp_ns,
                    frame_idx=self.frame_counter,
                    raw_data={'mouse': data, 'topic': topic}
                )
                self.observations[timestamp_ns] = frame
                self.frame_counter += 1
            else:
                for obs in self.observations.values():
                    if obs.timestamp_ns == timestamp_ns:
                        obs.raw_data['mouse'] = data
                        break
            
            # 提取鼠标位置信息
            if 'x' in data or 'position' in data:
                x = data.get('x', data.get('position', {}).get('x', 0))
                y = data.get('y', data.get('position', {}).get('y', 0))
                
                # 创建动作帧用于跟踪鼠标移动
                if self.last_mouse_pos != (x, y):
                    action = ActionFrame(
                        timestamp_ns=timestamp_ns,
                        action_idx=self.action_counter,
                        mouse_dx=x - self.last_mouse_pos[0],
                        mouse_dy=y - self.last_mouse_pos[1]
                    )
                    self.actions[timestamp_ns] = action
                    self.action_counter += 1
                    self.last_mouse_pos = (x, y)
            
            # 处理鼠标点击
            if 'button' in data or 'event_type' in data:
                event_type = data.get('event_type', '').lower()
                button = data.get('button', '').lower()
                
                if event_type == 'click' or 'click' in button:
                    action = ActionFrame(
                        timestamp_ns=timestamp_ns,
                        action_idx=self.action_counter,
                        mouse_click=button if button else 'left'
                    )
                    self.actions[timestamp_ns] = action
                    self.action_counter += 1
                    
        except Exception as e:
            logger.debug(f"处理鼠标消息时出错: {e}")
    
    def _process_keyboard_message(self, data: Dict, timestamp_ns: int, topic: str):
        """处理键盘消息"""
        try:
            # 创建或更新观察帧
            if timestamp_ns not in [obs.timestamp_ns for obs in self.observations.values()]:
                frame = ObservationFrame(
                    timestamp_ns=timestamp_ns,
                    frame_idx=self.frame_counter,
                    raw_data={'keyboard': data, 'topic': topic}
                )
                self.observations[timestamp_ns] = frame
                self.frame_counter += 1
            else:
                for obs in self.observations.values():
                    if obs.timestamp_ns == timestamp_ns:
                        if 'keyboard' not in obs.raw_data:
                            obs.raw_data['keyboard'] = {}
                        obs.raw_data['keyboard'].update(data)
                        break
            
            # 提取键盘事件
            event_type = data.get('event_type', 'unknown').lower()
            key = data.get('key', data.get('vk', ''))
            
            if event_type == 'press' or event_type == 'keydown':
                self.keyboard_state[key] = True
                action = ActionFrame(
                    timestamp_ns=timestamp_ns,
                    action_idx=self.action_counter,
                    keyboard_pressed=[key]
                )
                self.actions[timestamp_ns] = action
                self.action_counter += 1
            elif event_type == 'release' or event_type == 'keyup':
                self.keyboard_state[key] = False
                action = ActionFrame(
                    timestamp_ns=timestamp_ns,
                    action_idx=self.action_counter,
                    keyboard_released=[key]
                )
                self.actions[timestamp_ns] = action
                self.action_counter += 1
                
        except Exception as e:
            logger.debug(f"处理键盘消息时出错: {e}")
    
    def _save_screen_image(self, timestamp_ns: int, image_path: str):
        """保存屏幕图像到输出目录"""
        try:
            src_path = Path(image_path)
            if src_path.exists():
                dest_filename = f"frame_{timestamp_ns}.png"
                dest_path = self.images_dir / dest_filename
                
                # 如果是视频文件，提取指定时间戳的帧
                if src_path.suffix.lower() in ['.mp4', '.mkv', '.avi', '.mov']:
                    self._extract_video_frame(src_path, timestamp_ns, dest_path)
                else:
                    # 直接复制图像文件
                    cv2.imwrite(str(dest_path), cv2.imread(str(src_path)))
                
                self.screen_frame_cache[timestamp_ns] = dest_filename
                logger.debug(f"已保存屏幕图像: {dest_filename}")
                
        except Exception as e:
            logger.warning(f"保存屏幕图像时出错: {e}")
    
    def _process_base64_image(self, timestamp_ns: int, image_base64: str):
        """处理 Base64 编码的图像"""
        try:
            import base64
            
            image_data = base64.b64decode(image_base64)
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            dest_filename = f"frame_{timestamp_ns}.png"
            dest_path = self.images_dir / dest_filename
            cv2.imwrite(str(dest_path), img)
            
            self.screen_frame_cache[timestamp_ns] = dest_filename
            logger.debug(f"已保存 Base64 图像: {dest_filename}")
            
        except Exception as e:
            logger.warning(f"处理 Base64 图像时出错: {e}")
    
    def _extract_video_frame(self, video_path: Path, timestamp_ns: int, output_path: Path):
        """从视频文件提取指定时间戳的帧"""
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.warning(f"无法打开视频文件: {video_path}")
                return
            
            # 计算帧号
            fps = cap.get(cv2.CAP_PROP_FPS)
            time_sec = timestamp_ns / 1e9
            frame_no = int(time_sec * fps)
            
            # 跳转到该帧
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                cv2.imwrite(str(output_path), frame)
                logger.debug(f"已从视频提取帧: {output_path.name}")
            else:
                logger.warning(f"无法读取视频帧: {video_path} @ frame {frame_no}")
                
        except Exception as e:
            logger.warning(f"提取视频帧时出错: {e}")
    
    def _extract_video_frame_from_media_ref(self, timestamp_ns: int, video_path: str, pts_ns: int):
        """
        从 media_ref 中指定的视频文件提取帧
        
        Args:
            timestamp_ns: 观察的时间戳
            video_path: 视频文件相对路径
            pts_ns: 视频中的时间戳（纳秒）
        """
        try:
            # 尝试多种路径组合
            possible_paths = [
                Path(video_path),
                Path(self.mcap_path.parent) / video_path,
                Path('.') / video_path,
                Path('./mcap') / video_path,
            ]
            
            video_file = None
            for path in possible_paths:
                if path.exists():
                    video_file = path
                    break
            
            if not video_file:
                logger.debug(f"视频文件未找到: {video_path}")
                return
            
            cap = cv2.VideoCapture(str(video_file))
            if not cap.isOpened():
                logger.warning(f"无法打开视频文件: {video_file}")
                return
            
            # 使用 pts_ns 来获取视频中的帧
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 30  # 默认帧率
            
            # 将纳秒转换为秒，然后计算帧号
            time_sec = pts_ns / 1e9
            frame_no = int(time_sec * fps)
            
            # 跳转到该帧
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                dest_filename = f"frame_{timestamp_ns}.png"
                dest_path = self.images_dir / dest_filename
                cv2.imwrite(str(dest_path), frame)
                
                self.screen_frame_cache[timestamp_ns] = dest_filename
                logger.debug(f"已从视频提取帧: {dest_filename} (video: {video_file.name}, pts: {pts_ns}ns)")
            else:
                logger.debug(f"无法读取视频帧: {video_file} @ frame {frame_no}")
                
        except Exception as e:
            logger.debug(f"从 media_ref 提取视频帧时出错: {e}")
    
    def create_lerobot_dataset(self) -> Dict[str, Any]:
        """创建 LeRobotDataset 格式的数据集"""
        logger.info("创建 LeRobotDataset 格式数据集...")
        
        # 构建数据集元数据
        duration_ns = (self.last_timestamp or self.first_timestamp) - (self.first_timestamp or 0)
        duration_sec = duration_ns / 1e9
        
        dataset_info = {
            'name': self.dataset_name,
            'version': '1.0',
            'created_at': datetime.now().isoformat(),
            'source': str(self.mcap_path),
            'duration_seconds': duration_sec,
            'total_frames': len(self.observations),
            'total_actions': len(self.actions),
            'fps': self.video_fps,
            'first_timestamp_ns': self.first_timestamp,
            'last_timestamp_ns': self.last_timestamp,
            'channels': {
                'channels_count': len(self.channels_map),
                'channel_topics': [ch.topic for ch in self.channels_map.values()]
            }
        }
        
        # 构建观察数据
        observations_list = []
        for timestamp_ns in sorted(self.observations.keys()):
            obs = self.observations[timestamp_ns]
            
            # 添加屏幕图像路径
            if timestamp_ns in self.screen_frame_cache:
                obs.screen_image_path = str(self.screen_frame_cache[timestamp_ns])
            
            observations_list.append(obs.to_dict())
        
        # 构建动作数据
        actions_list = []
        for timestamp_ns in sorted(self.actions.keys()):
            action = self.actions[timestamp_ns]
            actions_list.append(action.to_dict())
        
        dataset = {
            'info': dataset_info,
            'observations': observations_list,
            'actions': actions_list,
            'raw_metadata': {
                'channels': {
                    str(ch_id): {
                        'topic': ch.topic,
                        'message_encoding': ch.message_encoding,
                        'schema_id': ch.schema_id
                    }
                    for ch_id, ch in self.channels_map.items()
                },
                'schemas': {
                    str(sch_id): {
                        'name': sch.name,
                        'encoding': sch.encoding
                    }
                    for sch_id, sch in self.schemas_map.items()
                }
            }
        }
        
        logger.info(f"数据集创建完成: {len(observations_list)} 个观察, {len(actions_list)} 个动作")
        return dataset
    
    def save_dataset(self, dataset: Dict[str, Any]) -> Path:
        """保存数据集到文件"""
        logger.info("保存数据集...")
        
        # 保存为 JSON
        dataset_path = self.output_dir / "dataset.json"
        with open(dataset_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        logger.info(f"数据集已保存: {dataset_path}")
        
        # 保存数据集摘要
        summary_path = self.output_dir / "summary.json"
        summary = {
            'dataset_name': dataset['info']['name'],
            'total_observations': len(dataset['observations']),
            'total_actions': len(dataset['actions']),
            'duration_seconds': dataset['info']['duration_seconds'],
            'fps': dataset['info']['fps'],
            'image_directory': str(self.images_dir.relative_to(self.output_dir)),
            'created_at': dataset['info']['created_at']
        }
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"数据集摘要已保存: {summary_path}")
        
        return dataset_path
    
    def convert(self) -> Tuple[Path, Dict[str, Any]]:
        """执行完整的转换流程"""
        logger.info("开始 MCAP 到 LeRobotDataset 转换")
        logger.info(f"MCAP 文件: {self.mcap_path}")
        logger.info(f"输出目录: {self.output_dir}")
        
        # 1. 解析 MCAP
        self.parse_mcap()
        
        # 2. 创建 LeRobotDataset
        dataset = self.create_lerobot_dataset()
        
        # 3. 保存数据集
        dataset_path = self.save_dataset(dataset)
        
        logger.info("转换完成!")
        return dataset_path, dataset


def main():
    """主函数示例"""
    import argparse
    
    parser = argparse.ArgumentParser(description='将 MCAP 数据转换为 LeRobotDataset 格式')
    parser.add_argument('mcap_file', help='MCAP 文件路径')
    parser.add_argument('-o', '--output', default='./lerobot_dataset', help='输出目录')
    parser.add_argument('-n', '--name', default='desktop_agent', help='数据集名称')
    parser.add_argument('--fps', type=int, default=30, help='视频帧率')
    parser.add_argument('--max-messages', type=int, default=None, help='最大消息数')
    
    args = parser.parse_args()
    
    # 创建转换器
    converter = MCAPToLeRobotConverter(
        mcap_path=args.mcap_file,
        output_dir=args.output,
        dataset_name=args.name,
        video_fps=args.fps,
        max_messages=args.max_messages
    )
    
    # 执行转换
    dataset_path, dataset = converter.convert()
    
    print(f"\n✅ 转换成功!")
    print(f"📁 数据集路径: {dataset_path}")
    print(f"📊 数据集信息:")
    print(f"   - 观察帧数: {len(dataset['observations'])}")
    print(f"   - 动作帧数: {len(dataset['actions'])}")
    print(f"   - 时长: {dataset['info']['duration_seconds']:.2f} 秒")
    print(f"📸 图像目录: {converter.images_dir}")


if __name__ == '__main__':
    main()
