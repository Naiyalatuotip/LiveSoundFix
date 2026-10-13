"""缓存清理规则测试：分类统计 / 容量与天数上限 / 保护集 / 不越界 / 内存 LRU。

全部在**临时目录**上做，绝不碰真实 cache/。
运行：python tests/test_cache.py
"""
import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from livefix import api  # noqa: E402

FAIL = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAIL.append(name)


tmp = tempfile.mkdtemp(prefix="lf_cache_test_")
outside = tempfile.mkdtemp(prefix="lf_cache_outside_")
try:
    # 造一批各类文件：p_/o_/stems_/转码/其它，大小与时间都可控
    def mk(name, size, age_days):
        p = os.path.join(tmp, name)
        with open(p, "wb") as f:
            f.write(b"\0" * size)
        t = time.time() - age_days * 86400
        os.utime(p, (t, t))
        return p

    names = []
    for i in range(12):
        names.append(mk(f"p_{i:08x}{0:0>0}.wav", 1 << 20, age_days=20 - i))       # 预览，越往后越新
    for i in range(5):
        names.append(mk(f"o_{i:08x}.wav", 1 << 20, age_days=15 - i))
    for i in range(4):
        names.append(mk(f"stems_job_{i:06x}.npz", 1 << 20, age_days=10 - i))
    names.append(mk("music_0123456789.wav", 1 << 20, age_days=30))               # 转码产物
    names.append(mk("_e2e_src.wav", 1 << 20, age_days=60))                       # 「其它」
    names.append(mk("readme.txt", 100, age_days=99))                             # 「其它」
    stray = os.path.join(outside, "p_deadbeef.wav")
    with open(stray, "wb") as f:
        f.write(b"\0" * 1024)

    # 1) 分类统计
    info = api.cache_info(cache_dir=tmp)
    g = info["groups"]
    print(f"[info] total={info['total']} groups="
          + ", ".join(f"{k}:{v['files']}" for k, v in g.items()))
    check("分类计数正确",
          g["preview"]["files"] == 12 and g["original"]["files"] == 5
          and g["stems"]["files"] == 4 and g["transcode"]["files"] == 1
          and g["other"]["files"] == 2,
          str({k: v["files"] for k, v in g.items()}))
    check("总字节正确（23 个 1MB + 100B）",
          info["total"]["bytes"] == 23 * (1 << 20) + 100, str(info["total"]["bytes"]))
    check("可删标记：other 不可删", g["other"]["deletable"] is False)

    # 2) dry_run 不删任何东西
    before = sorted(os.listdir(tmp))
    r = api.cache_clean(cache_dir=tmp, dry_run=True, max_mb=1)
    after = sorted(os.listdir(tmp))
    check("dry_run 报告了候选", r["ok"] and r["candidates"] > 0, f"候选 {r['candidates']}")
    check("dry_run 一个文件都没删", before == after and r["deleted"] == 0)

    # 3) keep_recent 保护：每类最新 N 个必须留下
    r = api.cache_clean(cache_dir=tmp, dry_run=False, max_mb=0, keep_recent=3)
    left = set(os.listdir(tmp))
    p_sorted = sorted([n for n in names if n.startswith("p_")],
                      key=lambda n: os.path.getmtime(os.path.join(tmp, n)), reverse=True)
    kept_newest = all(os.path.basename(p) in left for p in p_sorted[:3])
    check("每类最新 3 个被保护", kept_newest, f"留下 {len(left)} 个")
    check("清理真的删了文件", r["deleted"] > 0 and r["freed_bytes"] > 0,
          f"删 {r['deleted']} 个 / {r['freed_bytes']/1e6:.1f} MB")

    # 4) 「其它」类别默认不动
    check("其它类别默认不删", "_e2e_src.wav" in left and "readme.txt" in left)

    # 5) 不越界：临时目录外的同名文件必须原封不动
    check("未触碰目录外文件", os.path.exists(stray))

    # 6) 容量上限只能压到「保护集」为止：未被保护的可删文件必须清零
    r = api.cache_clean(cache_dir=tmp, dry_run=False, max_mb=2, keep_recent=1)
    info2 = api.cache_info(cache_dir=tmp, keep_recent=1)
    rec = info2["reclaimable"]
    left_dele = sum(v["files"] for k, v in info2["groups"].items() if v["deletable"])
    print(f"[cap] 剩余可删文件 {left_dele} 个 / 可回收 {rec['bytes']} B")
    check("容量上限：未被保护的可删文件已清零", rec["files"] == 0 and rec["bytes"] == 0,
          str(rec))
    check("剩下的恰是保护集（每类最新 1 个 × 4 类）", left_dele == 4, str(left_dele))

    # 7) 天数上限：造一个很旧的文件，只按天数删
    mk("p_ffffffff.wav", 4096, age_days=100)
    r = api.cache_clean(cache_dir=tmp, dry_run=False, max_age_days=30, keep_recent=0)
    check("天数上限删除 100 天前的文件",
          "p_ffffffff.wav" not in set(os.listdir(tmp)) and r["deleted"] >= 1,
          f"删 {r['deleted']} 个")

    # 8) include_other 时才动「其它」
    r = api.cache_clean(cache_dir=tmp, dry_run=False, max_mb=0, keep_recent=0,
                        include_other=True)
    check("include_other=True 时才清「其它」", "_e2e_src.wav" not in set(os.listdir(tmp)))

    # 9) 内存音频缓存是**有界 LRU**
    old_max, old_keep = api._AUDIO_MEM_MAX_BYTES, api._AUDIO_MEM_KEEP_MIN
    try:
        # 每条 1MB（float32 的 1<<18 个样本 = 1MB），上限 4MB、至少留 2 条
        api._AUDIO_MEM_MAX_BYTES, api._AUDIO_MEM_KEEP_MIN = 4 << 20, 2
        api._audio.clear()
        import numpy as np
        for i in range(10):
            api._audio_put(f"p_{i:08x}", np.zeros(1 << 18, np.float32), 48000)
        mem = api.audio_mem_info()
        print(f"[mem] entries={mem['entries']} bytes={mem['bytes']}")
        check("内存缓存被字节上限约束（≤4MB+1条）", mem["bytes"] <= (5 << 20), str(mem["bytes"]))
        check("至少保留 keep_min 条", mem["entries"] >= 2, str(mem["entries"]))
        check("淘汰的是最旧的（最新一条仍在）",
              list(api._audio.keys())[-1] == "p_00000009", str(list(api._audio.keys())))
    finally:
        api._AUDIO_MEM_MAX_BYTES, api._AUDIO_MEM_KEEP_MIN = old_max, old_keep
        api._audio.clear()
finally:
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.rmtree(outside, ignore_errors=True)

print("-" * 60)
if FAIL:
    print("CACHE TEST FAILED:", ", ".join(FAIL))
    sys.exit(1)
print("CACHE TEST OK")
