"""LiveFix AI 重建模块 —— 接口预留（本期不实现模型）。

设计意图（后续版本实现）：
- 独立训练程序（trainer/）在成对数据（劣化录音, 干净参考）上训练小型频谱掩码网络
  （如 2-3 层全卷积 / LSTMmask regressor，ONNX 导出，纯本地推理）。
- 本模块加载 livefix_models/*.npz（或未来的 .onnx），对 STFT 幅度谱预测 0..N 的
  增益掩码，focus 参数限定作用频段：low = 30-300Hz（低频重建）、vocal = 150-5000Hz
  （人声重建）；strength 控制掩码向 1.0 的收缩比例（strength=0 等于关闭）。
- 推理在 core.stft 域进行，保留原相位；模型缺失时按 rule 回退（谱包络平滑插值）。

当前行为：process 为透传，ctx["note"]["ai"] = "interface_only"，
engine 与 UI 均已按「接口预留」处理。
"""
from __future__ import annotations

import numpy as np

KEY = "ai"


def process(x, sr, p, ctx):
    """透传（接口预留）。未来实现：加载模型 → 预测掩码 → 应用到 |X| → istft。"""
    if ctx is not None:
        ctx["note"]["ai"] = "interface_only"
    return np.asarray(x, np.float32)


def info():
    return {"key": KEY, "name": "AI 重建", "algo": "接口预留（低频/人声频谱掩码模型）"}
