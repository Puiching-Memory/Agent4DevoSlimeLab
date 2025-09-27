import mss
import mss.tools
import pywinctl as pwc
import numpy as np
import time
import cv2
import os
import glob
import hashlib

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

def get_image_hash(img):
    """计算图像的MD5哈希值"""
    # 将图像转换为字节数据
    img_bytes = cv2.imencode('.png', img)[1].tobytes()
    # 计算MD5哈希值
    return hashlib.md5(img_bytes).hexdigest()

# 使用示例
# img = capture_window("DevoSlime Lab b0.4.64")

if __name__ == '__main__':
    directory = './raw_data_3'
    
    # 自动创建文件夹
    os.makedirs(directory, exist_ok=True)
    
    for i in range(25):
        img = capture_window("DevoSlime Lab b0.4.64")
        # 使用图像的哈希值作为文件名
        img_hash = get_image_hash(img)
        cv2.imwrite(f'{directory}/{img_hash}.png', img)
        print(f"Captured frame with hash: {img_hash}")
        time.sleep(3)