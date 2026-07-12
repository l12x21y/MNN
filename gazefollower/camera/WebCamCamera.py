#!/usr/bin/env python
# encoding=utf-8
# Author: GC Zhu
# Email: zhugc2016@gmail.com
import sys
import threading
import time
import traceback

import cv2

from .Camera import Camera  # Adjust import according to your package structure
from ..logger import Log


class WebCamCamera(Camera):
    """
    A class to manage webcam operations, inheriting from the base Camera class.
    """

    def __init__(self, webcam_id=0, img_height=480, img_width=640, cam_fps=30, process_fps=15):
        """
        Initializes the WebCamCamera object, sets up the camera properties,
        creates the capture thread, and ensures the save directory exists.

        Attributes:
        ----------
        webcam_id : int
            Which webcam camera is connected.
        cap: cv2.VideoCapture
            The instance of cv2.VideoCapture and it can be None.
        process_fps : int
            脸检测/眼动推理的最大频率(Hz)。摄像头线程仍按相机帧率读取最新帧,
            但只在距上次处理超过 1/process_fps 秒时才跑一次预处理+回调(face/gaze),
            把推理负载从相机帧率(约30Hz)降到约 process_fps(默认15Hz), 释放 CPU
            给主线程的视频解码与显示, 避免播放卡顿。该限速与睁眼度/眨眼剔除无关,
            不会影响校准质量。
        """
        super().__init__()
        self._camera_thread_running = None
        self._camera_thread = None
        self.webcam_id = webcam_id
        self.img_height = img_height
        self.img_width = img_width
        self.cam_fps = cam_fps
        self.process_fps = process_fps
        self._min_process_interval = 1.0 / max(process_fps, 1)
        self._last_process_ts = 0
        self._cap = cv2.VideoCapture()
        # Set the camera resolution and frame rate.
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.img_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.img_height)
        self._cap.set(cv2.CAP_PROP_FPS, self.cam_fps)

    def _create_capture_thread(self):
        """
        Creates and starts a daemon thread for continuously capturing frames from the camera.
        """
        self._camera_thread_running = True
        self._camera_thread = threading.Thread(target=self.capture)
        self._camera_thread.daemon = True
        self._camera_thread.start()

    def capture(self):
        """
        Continuously captures frames from the webcam while the camera is in specific running states.
        If a callback is set, it executes the callback function with the current frame.
        """
        while self._camera_thread_running:
            # Capture a frame from the webcam.
            ret, frame = self._cap.read()
            # Capture the current timestamp.
            timestamp = time.time_ns()
            if not ret:
                Log.w("Failed to grab frame")
                continue

            # 限速: 跳过过密的中间帧, 把脸检测/眼动推理降到约 process_fps Hz。
            # 仍在每个循环读取最新帧(OpenCV cap.read 直接返回当前帧, 不会积压),
            # 只是被跳过的帧不跑昂贵的预处理与回调, 从而释放 CPU 给视频解码/显示,
            # 避免播放卡顿。该跳过不影响睁眼度/眨眼剔除(校准质量保持当前状态)。
            if (timestamp - self._last_process_ts) / 1e9 < self._min_process_interval:
                continue
            self._last_process_ts = timestamp

            # Check if the frame is in BGR format (default for OpenCV) and convert to RGB if necessary
            # Preprocessing image data
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Resize the frame to 640x480 if necessary
            frame = cv2.resize(frame, (self.img_width, self.img_height))
            # Lock and execute callback function if set.
            try:
                with self.callback_and_param_lock:
                    if self.callback_func is not None:

                        self.callback_func(self.camera_running_state, timestamp, frame, *self.callback_args,
                                           **self.callback_kwargs)

            except Exception as e:
                Log.e(str(e))
                Log.e(f"Traceback:\n{traceback.format_exc()}")

    def open(self):
        """
        Opens the webcam if it is not already opened.
        """
        Log.i("WebCam opened")
        if not self._cap.open(self.webcam_id):
            Log.e("Failed to open webcam camera")
            Log.e(f"Traceback:\n{traceback.format_exc()}")
            raise Exception("Failed to open webcam camera")
        self._create_capture_thread()

    def close(self):
        """
        Releases the webcam resources if the camera is currently opened.
        """
        Log.i("WebCam closed")
        if self._camera_thread is not None:
            self._camera_thread_running = False
            self._camera_thread.join()
        # 无论是否已 open(), release() 在 OpenCV 中都是安全的; 原来的
        # "if not isOpened(): release()" 写反了, 会导致已打开的摄像头不被释放(泄漏)。
        self._cap.release()

    def set_on_image_callback(self, func, args=(), kwargs=None):
        """
        Sets a callback function to be called with each captured frame.
        The callback function must have the following args,
            timestamp and frame, which are the timestamp when the image was
            captured and the captured image frame (np.ndarray).

        Parameters:
        - func: The callback function to handle the image frame.
        - args: Tuple of arguments to pass to the callback function.
        - kwargs: Dictionary of keyword arguments to pass to the callback function.
        """
        super().set_on_image_callback(func, args, kwargs)

    def release(self):
        # 保护: 若摄像头从未 open()(线程未创建), 直接跳过 join, 避免
        # AttributeError: 'NoneType' object has no attribute 'join'。
        # 这样构造后直接 release / 初始化中途异常退出时也不会崩。
        if self._camera_thread is not None:
            self._camera_thread_running = False
            self._camera_thread.join()
        self.close()
