# encoding=utf-8
"""
快速检测 MNN 是否真正可用 GPU 后端(用于 GazeFollower 眼动追踪加速)。

用法(在已 pip install MNN 的项目虚拟环境里运行):
    python test_mnn_gpu.py

为什么需要"子进程"探测:
    官方 PyPI 的纯 CPU 版 MNN, 在请求 OpenCL(2)/CUDA(5)/Vulkan(4) 时不会报错,
    而是打印 "Can't Find type=2 backend, use 0 instead" 然后偷偷用 CPU 跑推理。
    更坑的是: 这条回退日志是 MNN 在【进程退出时】才打印的, 进程内 fd 重定向
    抓不到它, 所以普通探测会误报"可用 GPU"。

    本脚本用一个一次性的【子进程】探测所有后端, 父进程捕获子进程的完整输出
    (含退出时的日志), 再扫描 "Can't Find type=<b>", 从而如实报告 GPU 是否可用。

本脚本自包含(只依赖 MNN / numpy), 不 import gazefollower 包,
    以避免触发包初始化时要求的 Log.init()。
"""
import json
import subprocess
import sys
from pathlib import Path

MODEL = Path(__file__).parent / "gazefollower/res/model_weights/base.mnn"

# 子进程探测代码(只依赖 MNN / numpy, 必须自包含, 不能 import gazefollower 包)
_PROBE_CODE = r'''
import sys, json, gc
import numpy as np
import MNN
model = sys.argv[1]
backends = [2, 4, 5, 1, 0]
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

# (backend_id, 名称, 是否 GPU)
BACKENDS = [
    (2, "OpenCL (GPU)", True),
    (4, "Vulkan (GPU)", True),
    (5, "CUDA (GPU)", True),
    (1, "Metal (GPU, 仅 Mac)", True),
    (0, "CPU", False),
]

NAMES = {b: name for b, name, _ in BACKENDS}


def probe_all(model_path):
    """一次性子进程探测所有后端, 返回 {backend: {'ok': bool, 'fell_back': bool}}。"""
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
    for b in (2, 4, 5, 1, 0):
        ok = bool(res.get(str(b), False))
        fell_back = (b != 0) and ("Can't Find type=%d" % b in out)
        info[b] = {"ok": ok, "fell_back": fell_back}
    if 1 in info and sys.platform != "darwin":
        del info[1]
    return info, out


def main():
    if not MODEL.exists():
        print(f"[错误] 找不到模型文件: {MODEL}")
        sys.exit(1)
    try:
        import MNN
        import numpy as np
    except Exception as e:
        print("[结果] 未安装 MNN / numpy, 无法检测。请先安装:  pip install MNN numpy")
        print(f"        异常: {e}")
        sys.exit(2)

    print(f"Python:   {sys.executable}")
    print(f"MNN 版本: {getattr(MNN, '__version__', '未知')}")
    print(f"模型路径: {MODEL}")
    print("-" * 64)

    info, raw = probe_all(MODEL)

    for bid, name, is_gpu in BACKENDS:
        if bid not in info:
            continue
        d = info[bid]
        if d["ok"] and not d["fell_back"]:
            tag = "真 GPU" if is_gpu else "CPU"
            print(f"  [OK]   backend={bid:<2} {name}  -> 可用 ({tag})")
        elif d["ok"] and d["fell_back"]:
            print(f"  [伪GPU] backend={bid:<2} {name}  -> 推理跑通但被 MNN 静默回退到 CPU")
        else:
            print(f"  [FAIL] backend={bid:<2} {name}  -> 不可用")

    print("-" * 64)
    gpu_real = [b for b, d in info.items()
                if d["ok"] and not d["fell_back"] and NAMES.get(b, ("", "", False))[2]]
    if gpu_real:
        best = gpu_real[0]
        print(f"[结论] 可用 GPU! 推荐后端: backend={best} ({NAMES[best][0]})")
        print("        运行 demo 时无需任何设置(默认就会优先尝试 OpenCL/GPU),")
        print("        或显式指定后端更稳妥:")
        print("            Windows(cmd):  set GAZEFOLLOWER_MNN_BACKEND=%d" % best)
        print("            PowerShell:    $env:GAZEFOLLOWER_MNN_BACKEND=%d" % best)
    else:
        cpu_ok = info.get(0, {}).get("ok", False)
        if cpu_ok:
            print("[结论] 当前 MNN 不能用 GPU(已自动回退 CPU)。")
            print("        原因: 当前安装的 MNN 未包含 OpenCL/CUDA/Vulkan 等 GPU 后端,")
            print("        请求 GPU 时被静默回退到 CPU。需安装/编译带 GPU 后端的 MNN:")
            print("          - OpenCL: 系统需有 OpenCL ICD(随 Intel/AMD/NVIDIA 显卡驱动),")
            print("            并安装编译了 OpenCL 的 MNN(官方 PyPI wheel 多为纯 CPU)。")
            print("          - 或源码编译 MNN 并开启 MNN_OPENCL=ON 等选项。")
            print("        demo 已自动回退 CPU, 推理仍可用, 只是不吃 GPU 加速。")
        else:
            print("[结论] 所有后端(含 CPU)均不可用, 请检查 MNN 安装与模型文件。")
    print("-" * 64)


if __name__ == "__main__":
    main()
