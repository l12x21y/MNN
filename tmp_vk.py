# encoding=utf-8
"""一次性验证: backend=4 (Vulkan) 是否真正初始化了 GPU 设备(捕获全部 MNN 日志)。"""
import sys
import numpy as np
import MNN

model = "gazefollower/res/model_weights/base.mnn"
cfg = {'precision': 'low', 'backend': 4, 'numThread': 4}
print("=== 创建 Vulkan(4) runtime, 下面是 MNN 的完整原始输出 ===", flush=True)
rt = MNN.nn.create_runtime_manager((cfg,))
m = MNN.nn.load_module_from_file(
    model, ["face", "left", "right", "rect"], ["output_0"], runtime_manager=rt)
f = np.zeros((1, 224, 224, 3), np.float32)
l = np.zeros((1, 112, 112, 3), np.float32)
r = np.zeros((1, 112, 112, 3), np.float32)
c = np.zeros((1, 12), np.float32)
fv = MNN.expr.placeholder((1, 224, 224, 3), MNN.expr.NHWC)
lv = MNN.expr.placeholder((1, 112, 112, 3), MNN.expr.NHWC)
rv = MNN.expr.placeholder((1, 112, 112, 3), MNN.expr.NHWC)
cv = MNN.expr.placeholder((1, 12))
fv.write(f); lv.write(l); rv.write(r); cv.write(c)
o = m.onForward([fv, lv, rv, cv])
a = o[0].read()
print("=== forward 输出形状:", None if a is None else a.shape, "===")
