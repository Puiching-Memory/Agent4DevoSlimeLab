import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QTextEdit, QGroupBox, QTableWidget, QTableWidgetItem
)
from PyQt5.QtCore import Qt, QPoint, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QMouseEvent, QImage, QPixmap, QPainter, QPen, QColor

# 导入屏幕捕获和YOLO相关模块
import mss
from ultralytics import YOLO


class YOLOWorker(QThread):
    """YOLO识别工作线程"""
    result_ready = pyqtSignal(list)  # 识别结果信号
    
    def __init__(self, model, image):
        super().__init__()
        self.model = model
        self.image = image
        
    def run(self):
        """执行YOLO识别"""
        if self.model is not None and self.image is not None:
            results = self.model.predict(source=self.image, verbose=False)
            detection_boxes = []
            
            # 收集检测结果
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # 获取边界框坐标
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        class_name = self.model.names[class_id]
                        
                        # 保存检测框信息
                        detection_boxes.append({
                            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                            'confidence': confidence, 'class_id': class_id,
                            'class_name': class_name
                        })
            
            # 发送结果信号
            self.result_ready.emit(detection_boxes)


class ScreenCaptureWidget(QWidget):
    """屏幕捕获和YOLO识别结果展示窗口"""
    # 添加一个信号用于线程安全地更新检测结果
    detection_updated = pyqtSignal(list)
    
    def __init__(self):
        super().__init__()
        # 不设置透明背景，以便能看到绘制的内容
        # self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setWindowFlags(Qt.FramelessWindowHint)
        
        # 设置背景为半透明黑色，方便看到绘制的绿色框
        self.setStyleSheet("background-color: rgba(0, 0, 0, 30);")
        
        # 初始化变量
        self.detection_boxes = []  # 存储检测到的边界框
        self.last_capture_time = 0
        self.yolo_worker = None  # YOLO工作线程
        
        # 连接信号和槽
        self.detection_updated.connect(self.on_detection_updated)
        
        # 初始化YOLO模型
        try:
            self.model = YOLO("runs/slime_yolo/loop3/weights/best.pt")
        except Exception as e:
            print(f"无法加载YOLO模型: {e}")
            self.model = None
        
        # 设置定时器用于定期更新屏幕捕获
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_screen)
        self.timer.start(50)
        
    def update_screen(self):
        """更新屏幕捕获和YOLO识别结果"""
        # 获取当前窗口的位置和大小
        x, y = self.mapToGlobal(QPoint(0, 0)).x(), self.mapToGlobal(QPoint(0, 0)).y()
        w, h = self.width(), self.height()
        
        # 确保捕获区域有效
        if w <= 0 or h <= 0:
            return
        
        # 捕获屏幕区域
        img = self.capture_screen_region(x, y, w, h)
        
        if img is None:
            return
            
        # 保存原始图像用于预览
        self.original_image = img.copy()
            
        # 转换颜色空间 (BGR to RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        
        # 如果没有正在进行的YOLO识别任务，则启动新任务
        if self.yolo_worker is None or not self.yolo_worker.isRunning():
            # 清空之前的检测框
            self.detection_boxes = []
            self.update()  # 触发重绘以清除旧的检测框
            
            # 启动YOLO识别线程
            if self.model is not None:
                self.yolo_worker = YOLOWorker(self.model, img)
                self.yolo_worker.result_ready.connect(self.on_yolo_result)
                self.yolo_worker.start()
        
        # 触发重绘
        self.update()
        
    def on_yolo_result(self, detection_boxes):
        """处理YOLO识别结果"""
        # 使用信号来线程安全地更新检测结果
        self.detection_updated.emit(detection_boxes)
        
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
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 绘制窗口可见范围的边框
        pen = QPen(QColor(0, 255, 0))  # 绿色边框
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawRect(0, 0, self.width() - 1, self.height() - 1)
        
        
    def resizeEvent(self, event):
        """窗口大小改变事件"""
        super().resizeEvent(event)
        # 窗口大小改变时不直接触发屏幕捕获，而是等待定时器触发
        # 这样可以避免在调整窗口大小时产生递归调用
        
    def closeEvent(self, event):
        """窗口关闭事件"""
        # 等待YOLO工作线程完成
        if self.yolo_worker is not None and self.yolo_worker.isRunning():
            self.yolo_worker.quit()
            self.yolo_worker.wait()
        event.accept()
    
    def on_detection_updated(self, detection_boxes):
        """处理检测结果更新"""
        self.detection_boxes = detection_boxes
        self.update()  # 触发重绘


class TransparentWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("透明左侧区域示例")
        # 将窗口默认大小调整为1000x550
        self.setGeometry(100, 100, 1000, 550)
        
        # 设置窗口属性以支持透明并保持置顶
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        
        # 添加窗口拖拽功能所需的变量
        self.drag_position = QPoint()
        self.resizing = False
        self.resize_direction = None
        self.margin = 10
        
        # 创建中央窗口部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建主布局（水平布局）
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # 创建左侧透明区域
        self.create_left_panel()
        
        # 创建右侧文本显示控件区域
        self.create_right_panel()
        
        # 添加左右两侧到主布局，调整比例为3:1，增加透明区域
        main_layout.addWidget(self.left_panel, 3)
        main_layout.addWidget(self.right_panel, 1)
    
    def create_left_panel(self):
        """创建左侧透明面板"""
        self.left_panel = ScreenCaptureWidget()
        # 添加绿色边框以便识别边界
        self.left_panel.setStyleSheet("""
            QWidget {
                background-color: rgba(0, 0, 0, 30);
                border: 2px solid #00ff00;
                border-radius: 5px;
            }
        """)
    
    def create_right_panel(self):
        """创建右侧文本显示面板"""
        self.right_panel = QWidget()
        self.right_panel.setStyleSheet("""
            QWidget {
                background-color: white;
                border: 1px solid gray;
                border-radius: 5px;
            }
        """)
        
        layout = QVBoxLayout(self.right_panel)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # 添加标题
        title_label = QLabel("YOLO识别信息")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(title_label)
        
        # 添加截屏预览标签
        preview_label = QLabel("截屏预览:")
        preview_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        layout.addWidget(preview_label)
        
        # 添加截屏预览图像
        self.preview_image = QLabel()
        self.preview_image.setMinimumSize(200, 150)
        self.preview_image.setMaximumHeight(200)
        self.preview_image.setStyleSheet("border: 1px solid gray; background-color: #f0f0f0;")
        self.preview_image.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.preview_image)
        
        # 添加检测信息表格
        self.detection_table = QTableWidget()
        self.detection_table.setColumnCount(4)
        self.detection_table.setHorizontalHeaderLabels(["类别", "置信度", "位置(X,Y)", "大小(WxH)"])
        self.detection_table.setRowCount(0)
        layout.addWidget(self.detection_table)
        
        # 添加系统信息区域
        info_group = QGroupBox("系统信息")
        info_layout = QVBoxLayout()
        
        self.window_info = QLabel("窗口位置: (0, 0)\n窗口大小: 0 x 0")
        self.window_info.setStyleSheet("font-family: Consolas;")
        info_layout.addWidget(self.window_info)
        
        self.detection_info = QLabel("检测到的对象数量: 0")
        self.detection_info.setStyleSheet("font-family: Consolas;")
        info_layout.addWidget(self.detection_info)
        
        self.model_info = QLabel("模型状态: 未加载")
        self.model_info.setStyleSheet("font-family: Consolas;")
        info_layout.addWidget(self.model_info)
        
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        # 定时更新右侧信息
        self.info_timer = QTimer(self)
        self.info_timer.timeout.connect(self.update_info_display)
        self.info_timer.start(40)
    
    def update_info_display(self):
        """更新右侧信息显示"""
        try:
            # 更新窗口信息
            if hasattr(self, 'left_panel'):
                # 获取透明窗口（主窗口）的位置和大小
                x, y = self.x(), self.y()
                w, h = self.width(), self.height()
                self.window_info.setText(f"窗口位置: ({x}, {y})\n窗口大小: {w} x {h}")
                
                # 更新检测信息
                detection_count = 0
                if hasattr(self.left_panel, 'detection_boxes'):
                    detection_count = len(self.left_panel.detection_boxes)
                self.detection_info.setText(f"检测到的对象数量: {detection_count}")
                
                # 更新模型信息
                model_status = "未加载"
                if hasattr(self.left_panel, 'model') and self.left_panel.model is not None:
                    model_status = "已加载"
                self.model_info.setText(f"模型状态: {model_status}")
                    
                # 更新检测表格
                self.detection_table.setRowCount(detection_count)
                if hasattr(self.left_panel, 'detection_boxes'):
                    for i, box in enumerate(self.left_panel.detection_boxes):
                        # 类别
                        class_item = QTableWidgetItem(box.get('class_name', 'Unknown'))
                        class_item.setFlags(class_item.flags() & ~Qt.ItemIsEditable)
                        self.detection_table.setItem(i, 0, class_item)
                        
                        # 置信度
                        conf_item = QTableWidgetItem(f"{box.get('confidence', 0):.3f}")
                        conf_item.setFlags(conf_item.flags() & ~Qt.ItemIsEditable)
                        self.detection_table.setItem(i, 1, conf_item)
                        
                        # 位置
                        pos_item = QTableWidgetItem(f"({box.get('x1', 0)}, {box.get('y1', 0)})")
                        pos_item.setFlags(pos_item.flags() & ~Qt.ItemIsEditable)
                        self.detection_table.setItem(i, 2, pos_item)
                        
                        # 大小
                        width = box.get('x2', 0) - box.get('x1', 0)
                        height = box.get('y2', 0) - box.get('y1', 0)
                        size_item = QTableWidgetItem(f"{width} x {height}")
                        size_item.setFlags(size_item.flags() & ~Qt.ItemIsEditable)
                        self.detection_table.setItem(i, 3, size_item)
                
                # 调整列宽
                self.detection_table.resizeColumnsToContents()
                
                # 更新预览图像
                if hasattr(self.left_panel, 'original_image'):
                    img = self.left_panel.original_image.copy()  # 创建副本以避免修改原始图像
                    # 转换颜色空间 (BGRA to RGB)
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                    
                    # 在预览图像上绘制检测框
                    if hasattr(self.left_panel, 'detection_boxes'):
                        for box in self.left_panel.detection_boxes:
                            # 获取边界框坐标
                            x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
                            
                            # 绘制边界框
                            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            
                            # 绘制标签
                            label = f"{box['class_name']} {box['confidence']:.2f}"
                            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                    # 转换颜色空间以适应Qt
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    h, w, ch = img.shape
                    bytes_per_line = ch * w
                    q_img = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(q_img)
                    # 缩放图像以适应预览区域
                    self.preview_image.setPixmap(pixmap.scaled(
                        self.preview_image.width(), 
                        self.preview_image.height(),
                        Qt.KeepAspectRatio, 
                        Qt.SmoothTransformation))
        except Exception as e:
            # 忽略更新错误，避免中断定时器
            pass
    
    def mousePressEvent(self, event: QMouseEvent):
        """鼠标按下事件"""
        if event.button() == Qt.LeftButton:
            self.drag_position = event.globalPos() - self.frameGeometry().topLeft()
            
            # 检查是否在边缘，用于调整窗口大小
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
                # 调整窗口大小
                self._resize_window(event.globalPos())
            elif not self.resizing:
                # 移动窗口
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
        # 关闭左侧面板（会触发其中的线程清理）
        if hasattr(self, 'left_panel'):
            self.left_panel.close()
        event.accept()


def main():
    app = QApplication(sys.argv)
    
    # 设置应用程序样式
    app.setStyle('Fusion')
    
    window = TransparentWindow()
    window.show()
    
    # 确保应用程序正常退出
    exit_code = app.exec_()
    sys.exit(exit_code)


if __name__ == '__main__':
    main()