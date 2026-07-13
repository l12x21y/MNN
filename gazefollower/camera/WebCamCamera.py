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
            摄像头线程每 1/process_fps 秒才跑一次脸检测+眼动推理并写一条样本。
            该频率直接决定录制 CSV 的采样密度, 也是播放期 CPU/GIL 占用的主要来源。
            默认 15Hz（校准用）；录制前可调用 set_process_fps() 调低到 10Hz 等。
        """
        super().__init__()
        self._camera_thread_running = None
        self._camera_thread = None
        self.webcam_id = webcam_id
        self.img_height = img_height
        self.img_width = img_width
        self.cam_fps = cam_fps
        self.process_fps = process_fps
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

        频率控制: 每个处理周期后 sleep 以维持 process_fps, 从而控制录制采样密度,
        并减少 CPU/GIL 占用（尤其播放视频期间, 留给主线程解码/显示视频更多算力）。
        """
        while self._camera_thread_running:
            t0 = time.time()
            # Capture a frame from the webcam.
            ret, frame = self._cap.read()
            # Capture the current timestamp.
            timestamp = time.time_ns()
            if not ret:
                Log.w("Failed to grab frame")
                continue

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

            # 频率控制: 维持 process_fps（处理一帧的耗时也算在内）
            if self.process_fps and self.process_fps > 0:
                dt = time.time() - t0
                sleep_t = (1.0 / self.process_fps) - dt
                if sleep_t > 0:
                    time.sleep(sleep_t)
                # 让出 GIL, 减轻与播放视频主线程的竞争
                time.sleep(0)

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

    def set_process_fps(self, fps: int):
        """
        设置摄像头线程的处理频率(Hz)。

        该频率控制每 1/fps 秒跑一次脸检测+眼动推理并写一条样本, 即直接决定
        录制 CSV 的采样密度。录制前调低(demo 中 RECORD_HZ=10)可释放更多 CPU
        给视频播放, 保证流畅; 校准时可保持默认的较高值(15Hz)以更快采集样本。
        """
        self.process_fps = max(1, int(fps))

    def close(self):
        """
        Releases the webcam resources if the camera is currently opened.
        """
        Log.i("WebCam closed")
        if self._camera_thread is not None:
            self._camera_thread_running = False
            self._camera_thread.join()
        if not self._cap.isOpened():
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
        self._camera_thread_running = False
        self._camera_thread.join()
        self.close()
