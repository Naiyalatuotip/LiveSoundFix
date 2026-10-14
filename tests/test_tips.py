"""解释框（UI 参数悬浮说明）自检。

守住三件事（2026-10-14 用户要求）：
  1. 每个挂了 data-tip 的控件都能在 TIPS 表里查到正文 —— 旧版把键名当内容渲染，
     所有解释框只显示 "m1.weight" 这种字样（TIPS 表整张表是死代码）。
  2. 正文三段式：参数名(key) / 频段·操作方式 / 目的·预估效果。
  3. 正文长度约 40 字（区间 28–48），保证既有信息量又不冗长。

运行：python tests/test_tips.py
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HTML_PATH = os.path.join(ROOT, "web", "index.html")
FAIL = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAIL.append(name)


html = open(HTML_PATH, encoding="utf-8").read()

# ---- 1) 解析 TIPS 表 ----
m = re.search(r"const TIPS = \{(.*?)\n\};", html, re.S)
check("能定位 TIPS 表", bool(m))
if not m:
    print("TIPS 表缺失，终止"); sys.exit(1)
tips = {}
for km, vm in re.findall(r'"([^"]+)":\s*"((?:[^"\\]|\\.)*)"', m.group(1)):
    tips[km] = vm.replace("\\n", "\n")
print(f"       TIPS 条目 {len(tips)}")

# ---- 2) 解析 MODS 参数表，推导所有会被使用的键 ----
mb = re.search(r"const MODS = \{(.*?)\n\};", html, re.S)
check("能定位 MODS 表", bool(mb))
mod_params, cur = {}, None
for line in (mb.group(1) if mb else "").split("\n"):
    mm = re.match(r"\s{2}(\w+): \{ name:", line)
    if mm:
        cur = mm.group(1); mod_params[cur] = []; continue
    pk = re.search(r'\{ k: "(\w+)"', line)
    if pk and cur:
        mod_params[cur].append(pk.group(1))
chain = re.findall(r'"(\w+)"', re.search(r"const CHAIN = \[([^\]]+)\]", html).group(1))

expected = set()
for k in chain:
    expected.add(f"mod.{k}")          # 模块卡片标题
    expected.add(f"mix.{k}")          # 干湿比 9 行
    if k not in ("ai", "dyneq"):      # 调音台推子（ai/dyneq 不在推子列表）
        expected.add(f"mx.{k}")
for k, ps in mod_params.items():
    for p in ps:
        expected.add(f"{k}.{p}")
for km in re.findall(r'data-tip="([a-zA-Z]+\.[a-z_]+)"', html):   # 字面量（eq/exp/prog/dyneq）
    expected.add(km)

missing = sorted(k for k in expected if k not in tips)
check(f"所有 data-tip 键都能查到正文（期望 {len(expected)} 个）", not missing,
      f"缺：{missing}" if missing else "不再显示键名")

unused = sorted(k for k in tips if k not in expected)
check("TIPS 表没有死条目", not unused, f"未使用：{unused}" if unused else "")

# ---- 3) 三段式 ----
bad3 = [k for k, v in tips.items() if len(v.split("\n")) != 3]
check("全部为三段式（标题/频段·操作/目的·效果）", not bad3, f"非三段：{bad3}")

empty = [k for k, v in tips.items() if any(len(x.strip()) == 0 for x in v.split("\n"))]
check("没有空行段", not empty, str(empty))

# ---- 4) 正文长度约 40 字 ----
lens = {k: len("".join(v.split("\n")[1:])) for k, v in tips.items()}
short = {k: n for k, n in lens.items() if n < 28}
long_ = {k: n for k, n in lens.items() if n > 48}
vals = sorted(lens.values())
print(f"       正文长度：最短 {vals[0]} / 中位 {vals[len(vals)//2]} / 最长 {vals[-1]}")
check("正文都 ≥28 字（信息量足够）", not short, str(short))
check("正文都 ≤48 字（不冗长）", not long_, str(long_))
check("正文中位落在 32–44 字（约 40 字）", 32 <= vals[len(vals) // 2] <= 44,
      f"中位 {vals[len(vals)//2]}")

# ---- 5) 频段 / 作用范围：每类控件都要说清「作用于什么」
#   音频处理类参数必须点明频段；模式 / 导出 / 进度 / 干湿比 / 推子类控件没有频段，
#   改为要求说明作用对象（全频段、整链、推子、声道、位深、阶段…）。
BAND_RE = re.compile(
    r"\d+\s?(Hz|kHz)|全频|频段|频点|频谱|带宽|全带|宽带|低频|中低|中高|中频|高频|超低")
SCOPE_RE = re.compile(
    r"\d+\s?(Hz|kHz)|全频|频段|频点|频谱|带宽|全带|宽带|低频|中低|中高|中频|高频|超低"
    r"|整链|全链|处理链|阶段|推子|贡献量|声道|位深|模块|干湿比|音频|作用")
BAND_REQUIRED = re.compile(r"^(m[1-6]|sep|dyneq|eq)\.")          # 频段类参数
BAND_REQUIRED_MOD = re.compile(r"^mod\.(sep|m[1-6]|dyneq)$")    # 频段类模块卡片

miss_band, miss_scope = [], []
for k, v in tips.items():
    line2 = v.split("\n")[1]
    if BAND_REQUIRED.match(k) or BAND_REQUIRED_MOD.match(k):
        if not BAND_RE.search(line2):
            miss_band.append(k)
    else:
        if not SCOPE_RE.search(line2):
            miss_scope.append(k)
check("频段类参数的第二行都标出频段", not miss_band, f"缺：{miss_band}")
check("其余控件的第二行都说明作用范围", not miss_scope, f"缺：{miss_scope}")

print("-" * 60)
if FAIL:
    print("TIPS TEST FAILED:", ", ".join(FAIL))
    sys.exit(1)
print("TIPS TEST OK")
