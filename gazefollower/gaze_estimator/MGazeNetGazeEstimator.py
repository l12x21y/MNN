# encoding=utf-8
# Author: GC Zhu
# Email: zhugc2016@gmail.com

import os
import pathlib
import traceback

import MNN
import cv2
import numpy as np

from ..gaze_estimator import GazeEstimator
from ..logger import Log
from ..misc import FaceInfo, GazeInfo, TrackingState, clip_patch
from ..mnn_probe import probe_all, load_module


class MGazeNetGazeEstimator(GazeEstimator):
    """
    A gaze estimation class that utilizes a pre-trained MNN model with MNN Module API.

    This class is responsible for loading the model, preparing input data,
    and running inference to estimate gaze direction based on facial information.
    """

    def __init__(self, model_path: str = ""):
        """
        Initialize the MGazeNetGazeEstimator.

        Loads the model weights and sets up the interpreter and session for inference.

        后端自动选择: 优先 GPU(OpenCL=2, 否则 Vulkan=4), 该后端能通过"自检前向"且未
        被 MNN 静默回退才使用, 否则回退 CPU(backend=0)。可用环境变量
        GAZEFOLLOWER_MNN_BACKEND 强制指定 (0=CPU, 2=OpenCL, 4=Vulkan); 若指定的后端
        不可用, 仍会自动回退到 CPU, 保证可用。
        """
        super().__init__()

        # Load model weights from the specified path
        if model_path == "":
            self.model_path = pathlib.Path(__file__).parent.parent / "res/model_weights/base.mnn"
        else:
            self.model_path = pathlib.Path(model_path).resolve()

        # 输入张量尺寸(需在加载前定义, 供自检使用)
        self.face_input_format = (1, 224, 224, 3)
        self.eye_input_format = (1, 112, 112, 3)
        self.rect_input_format = (1, 12)

        # 自动选择后端: 优先 GPU(OpenCL=2, 其次 Vulkan=4), 失败回退 CPU(0)
        # 先一次性子进程探测所有后端(probe_all 能识别 MNN "静默回退到 CPU" ——
        # 部分纯 CPU 版 MNN 请求 OpenCL 时不报错, 只打印
        # "Can't Find type=2 backend, use 0 instead" 然后偷偷用 CPU;
        # 同理 Vulkan(4) 在缺少 ICD 时也会被静默回退),
        # 再对选中的后端用 load_module 在进程内正式加载, 避免误报"正在使用 GPU"。
        env_b = os.environ.get("GAZEFOLLOWER_MNN_BACKEND")
        if env_b is not None:
            candidates = [int(env_b)]
            if int(env_b) != 0:
                candidates.append(0)  # 强制指定非 CPU 后端失败时, 仍回退 CPU
        else:
            candidates = [2, 4, 0]   # OpenCL(GPU) -> Vulkan(GPU) -> CPU

        probe = probe_all(self.model_path)
        self.backend = None
        last_err = None
        for backend in candidates:
            d = probe.get(backend)
            if not d or not d["ok"] or d["fell_back"]:
                last_err = (
                    f"backend={backend} 探测未通过"
                    + ("(被 MNN 静默回退到 CPU)" if d and d["fell_back"] else "")
                )
                Log.w(f"MNN 后端 {backend} 不可用, 尝试下一个: {last_err}")
                continue
            try:
                self.gaze_module = load_module(backend, self.model_path)
                self.backend = backend
                Log.i(f"MGazeNet 使用后端 backend={backend} "
                      f"({'GPU' if backend != 0 else 'CPU'})")
                break
            except Exception as e:
                last_err = e
                Log.w(f"MNN 后端 {backend} 加载失败, 尝试下一个: {e}")
        if self.backend is None:
            raise RuntimeError(f"MNN 所有候选后端均初始化失败: {last_err}")

        # Size definitions for face and eye patches
        self.face_size = (224, 224)
        self.eye_size = (112, 112)
        self.rect_size = (1, 12)

        # Placeholder variables for input (will be updated in detect method)
        self.face_var = MNN.expr.placeholder(self.face_input_format, MNN.expr.NHWC)
        self.left_var = MNN.expr.placeholder(self.eye_input_format, MNN.expr.NHWC)
        self.right_var = MNN.expr.placeholder(self.eye_input_format, MNN.expr.NHWC)
        self.rect_var = MNN.expr.placeholder(self.rect_input_format)

    def detect(self, image, face_info: FaceInfo) -> GazeInfo:
        """
        Detect gaze direction from the input image and face information.

        :param image: The input image from which to estimate gaze direction.
        :param face_info: An instance of FaceInfo containing information about the detected face.
        :return: An instance of GazeInfo containing estimated gaze direction and related features.
        """
        gaze_info = GazeInfo()
        gaze_info.timestamp = face_info.timestamp

        # Check if face is detected
        if not face_info.status:
            gaze_info.tracking_state = TrackingState.FACE_MISSING
            return gaze_info

        # Check if gaze estimation is possible
        if not face_info.can_gaze_estimation:
            gaze_info.tracking_state = TrackingState.OUT_OF_BOUNDARIES
            return gaze_info

        # Extract face and eye rectangles
        f_x, f_y, f_w, f_h = face_info.face_rect
        le_x, le_y, le_w, le_h = face_info.left_rect
        re_x, re_y, re_w, re_h = face_info.right_rect

        # Clip patches for face and eyes
        face_patch = clip_patch(image, face_info.face_rect)
        left_eye_patch = clip_patch(image, face_info.left_rect)
        right_eye_patch = clip_patch(image, face_info.right_rect)

        # Check if any patch is None
        if face_patch is None or left_eye_patch is None or right_eye_patch is None:
            gaze_info.tracking_state = TrackingState.OUT_OF_BOUNDARIES
            return gaze_info

        # Prepare the rectangle input tensor with normalized values
        rect = (np.array([f_w, f_h, f_x, f_y,
                          le_w, le_h, le_x, le_y,
                          re_w, re_h, re_x, re_y], dtype=np.float32) /
                np.array(([face_info.img_w, face_info.img_h] * 6), dtype=np.float32))

        # Reshape rect to match input format
        rect = rect.reshape(1, 12)

        # Resize and normalize the patches for input
        face_patch_resized = cv2.resize(face_patch, self.face_size).astype(np.float32) / 255.0
        left_patch_resized = cv2.resize(left_eye_patch, self.eye_size).astype(np.float32) / 255.0
        right_patch_resized = cv2.resize(right_eye_patch, self.eye_size).astype(np.float32) / 255.0

        # Flip the right eye patch horizontally for correct gaze estimation
        right_patch_resized = cv2.flip(right_patch_resized, 1)

        try:
            # Write data to input variables
            self.face_var.write(face_patch_resized)
            self.left_var.write(left_patch_resized)
            self.right_var.write(right_patch_resized)
            self.rect_var.write(rect)

            # Prepare inputs for the module
            inputs = [self.face_var, self.left_var, self.right_var, self.rect_var]

            # Run inference
            outputs = self.gaze_module.onForward(inputs)

            if len(outputs) > 0:
                output_tensor = outputs[0]
                res = output_tensor.read()
                res = res.copy()
                res = res.reshape(-1).astype(np.float32)
                gaze_info.features = res
                gaze_info.raw_gaze_coordinates = res[:2]  # Extract gaze coordinates
                gaze_info.status = True
                gaze_info.left_openness = face_info.left_eye_openness
                gaze_info.right_openness = face_info.right_eye_openness
                gaze_info.tracking_state = TrackingState.SUCCESS
            else:
                gaze_info.status = False
                gaze_info.tracking_state = TrackingState.FAILURE

        except Exception as e:
            Log.e(e)
            Log.e(traceback.format_exc())
            gaze_info.status = False
            gaze_info.tracking_state = TrackingState.FAILURE

        return gaze_info

    def release(self):
        """
        Release resources.
        """
        # MNN Module doesn't require explicit release in this case
        pass
