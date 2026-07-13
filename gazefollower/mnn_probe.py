# encoding=utf-8
"""
MNN 后端可用性探测: 创建运行时 + 加载模型 + 自检前向, 并识别 MNN 的"静默回退到 CPU"。

背景: 部分 MNN 构建(尤其是官方 PyPI 纯 CPU wheel)在请求 OpenCL(2)/CUDA(5) 时,
create_runtime_manager / load_module_from_file 不会抛异常, 而是打印
"Can't Find type=2 backend, use 0 instead" 并悄悄用 CPU 跑推理。这种"伪 GPU"前端
会让眼动追踪在没有任何报错的情况下丢失 GPU 加速, 还容易误报"在用 GPU"。

难点: 该回退日志是 MNN 在【进程退出时】才打印的, 用进程内 fd 重定向无法在后端选择
阶段抓到它。因此这里用一次性【子进程】探测所有后端: 父进程捕获子进程的完整输出
(含退出时的日志), 从而可靠识别静默回退。

用法:
    from gazefollower.mnn_probe import probe_all, load_module
    info = probe_all(model_path)          # {2:{'ok':..,'fell_back':..}, 0:{...}, ...}
    mod  = load_module(backend, model_path)  # 在进程内正式加载(确认可用后)
"""
import gc
import json
import sys
import subprocess

import numpy as np
import MNN


# 子进程探测代码(只依赖 MNN / numpy, 必须自包含, 不能 import gazefollower 包)
_PROBE_CODE = r'''
import sys, json, gc
import numpy as np
import MNN
model = sys.argv[1]
backends = [6, 2, 4, 5, 1, 0]  # CUDA(6) → OpenCL(2) → Vulkan(4) → Metal(5) → OpenCL-macOS(1) → CPU(0)
if sys.platform != "darwin":
    backends = [b for b in backends if b != 1]
res = {}
for b in backends:
    try:
        cfg = {'precision': 'low', 'backend': b, 'numThread': 4}
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
        res[str(b)] = (a is not None and int(getattr(a, "size", 0)) > 0)
    except Exception:
        res[str(b)] = False
    try:
        del m
    except Exception:
        pass
    try:
        del rt
    except Exception:
        pass
    gc.collect()
print("PROBE_RESULTS=" + json.dumps(res))
'''

FACE_FMT = (1, 224, 224, 3)
EYE_FMT = (1, 112, 112, 3)
RECT_FMT = (1, 12)


def self_test_module(module):
    """用全零输入跑一次前向, 验证该后端(或回退后的 CPU)真的能推理。"""
    face = np.zeros(FACE_FMT, dtype=np.float32)
    left = np.zeros(EYE_FMT, dtype=np.float32)
    right = np.zeros(EYE_FMT, dtype=np.float32)
    rect = np.zeros(RECT_FMT, dtype=np.float32)
    fv = MNN.expr.placeholder(FACE_FMT, MNN.expr.NHWC)
    lv = MNN.expr.placeholder(EYE_FMT, MNN.expr.NHWC)
    rv = MNN.expr.placeholder(EYE_FMT, MNN.expr.NHWC)
    cv = MNN.expr.placeholder(RECT_FMT)
    fv.write(face)
    lv.write(left)
    rv.write(right)
    cv.write(rect)
    outs = module.onForward([fv, lv, rv, cv])
    if not outs or outs[0] is None:
        raise RuntimeError("self-test forward 无输出")
    arr = outs[0].read()
    if arr is None or int(getattr(arr, "size", 0)) == 0:
        raise RuntimeError("self-test forward 输出为空")


def probe_all(model_path):
    """一次性子进程探测所有后端, 返回 {backend: {'ok': bool, 'fell_back': bool}}。

    'fell_back' 表示该 GPU 后端被 MNN 静默回退到了 CPU(日志出现
    "Can't Find type=<b>")。CPU(0) 不会 fell_back。
    """
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE_CODE, str(model_path)],
        capture_output=True, text=True, timeout=180,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    res = {}
    for line in out.splitlines():
        if line.startswith("PROBE_RESULTS="):
            try:
                res = json.loads(line[len("PROBE_RESULTS="):])
            except Exception:
                res = {}
            break
    info = {}
    for b in (6, 2, 4, 5, 1, 0):
        ok = bool(res.get(str(b), False))
        fell_back = (b != 0) and ("Can't Find type=%d" % b in out)
        info[b] = {"ok": ok, "fell_back": fell_back}
    if 1 in info and sys.platform != "darwin":
        del info[1]
    # macOS 没有 CUDA (6), 移除以避免误报
    if sys.platform == "darwin" and 6 in info:
        del info[6]
    return info


def load_module(backend, model_path, num_thread=4):
    """在进程内正式加载指定后端的模型并自检; 失败抛异常。"""
    config = {'precision': 'low', 'backend': backend, 'numThread': num_thread}
    rt = MNN.nn.create_runtime_manager((config,))
    module = MNN.nn.load_module_from_file(
        str(model_path),
        ["face", "left", "right", "rect"],
        ["output_0"],
        runtime_manager=rt,
    )
    self_test_module(module)
    return module
