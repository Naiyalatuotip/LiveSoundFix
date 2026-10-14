"""端到端终验：真实 server + 真实内核，走 job → remix → export 全闭环。

自起 server.py 子进程（独立端口，无 fake api），全部走 HTTP。
"""
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
PORT = 8891
BASE = f"http://127.0.0.1:{PORT}"
SRC = os.path.join(ROOT, "cache", "_e2e_src.wav")


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=30) as r:
        return r.status, r.read()


def get_json(path):
    st, body = get(path)
    return st, json.loads(body)


def post(path, obj):
    req = urllib.request.Request(BASE + path, data=json.dumps(obj).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.status, json.loads(r.read())


def post_raw(path, blob):
    """拖拽导入用的原始字节流 POST（不是 JSON）。"""
    req = urllib.request.Request(BASE + path, data=blob,
                                 headers={"Content-Type": "application/octet-stream"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:      # 业务错误用 200，这里兜底非 2xx
        return e.code, json.loads(e.read() or b"{}")


def ensure_fixture():
    """仓库不含训练/测试素材，缺就现场合成一个 10s 立体声样本，保证新克隆也能跑。"""
    if os.path.exists(SRC):
        return
    sys.path.insert(0, ROOT)
    import numpy as np
    from livefix import wavio
    sr = 48000
    rng = np.random.default_rng(7)
    t = np.arange(10 * sr) / sr
    L = (0.3 * np.sin(2 * np.pi * 110 * t) + 0.2 * np.sin(2 * np.pi * 1200 * t)
         + 0.05 * rng.standard_normal(t.shape))
    R = L + 0.02 * rng.standard_normal(t.shape)
    x = np.stack([L, R], 1).astype(np.float32)
    os.makedirs(os.path.dirname(SRC), exist_ok=True)
    wavio.write_wav(SRC, x, sr, 24)
    print(f"[fixture] 缺测试素材，已现场生成：{SRC}")
    sys.path.pop(0)


def main():
    ensure_fixture()
    env = dict(os.environ)
    env.pop("LIVEFIX_FAKE_API", None)  # 强制真实内核
    proc = subprocess.Popen([PY, "server.py", "--port", str(PORT)], cwd=ROOT, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    results = []

    def check(name, cond, detail=""):
        results.append((name, bool(cond)))
        print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")

    try:
        # 等启动
        for _ in range(60):
            try:
                st, _b = get("/api/state")
                break
            except Exception:  # noqa: BLE001
                time.sleep(0.5)
        else:
            raise RuntimeError("server 未在 30s 内启动")

        # 1) 三区布局 markup
        st, body = get("/")
        html = body.decode("utf-8", "ignore")
        check("GET / 200", st == 200)
        for marker in ("grid-template-columns", "createAnalyser", "getFloatFrequencyData",
                       "BiquadFilter", "mixerHost", "eqCanvas", "/api/remix"):
            check(f"  含 {marker}", marker in html)
        check("  无外网 URL", ("http://" not in html.replace("http://127.0.0.1", "")
                             and "https://" not in html))

        # 2) state
        st, s = get_json("/api/state")
        mods = s["config"]["modules"]
        check("state: 9 模块含 mix", len(mods) == 9 and all("mix" in v for v in mods.values()))

        # 3) job（10 秒窗口 + 干湿比改动，验证 mix 进链）
        mods["m4"]["mix"] = 60
        st, r = post("/api/job", {"path": SRC, "config": s["config"], "window": [0, 10]})
        check("POST /api/job", r.get("ok") and r.get("job"), str(r)[:80])
        jid = r["job"]
        t0, last_frac, saw_eta, stages_seen = time.time(), -1, False, set()
        while True:
            st, j = get_json(f"/api/job?id={jid}")
            assert j.get("ok"), j
            if j["state"] != "running":
                break
            check("  frac 单调", j["frac"] >= last_frac - 1e-9 if last_frac >= 0 else True)
            last_frac = j["frac"]
            if j.get("eta_s") is not None:
                saw_eta = True
            stages_seen.add(j["stage"])
            assert time.time() - t0 < 180, "job 超时"
            time.sleep(0.4)
        check("job done", j["state"] == "done", f"err={j['error']}")
        check("job 过程出过 ETA", saw_eta)
        res = j["result"]
        check("  result.windowed", res["windowed"] and res["duration"] == 10.0)
        check("  stages 含 mix=60", any(s["key"] == "m4" and s["mix"] == 60
                                       for s in res["stats"]["stages"]),
              str([(x['key'], x.get('mix')) for x in res['stats']['stages']]))
        check("  mixer 可用", res["mixer"]["available"] and res["mixer"]["job"] == jid,
              str(res["mixer"]))

        # 4) audio 可播
        st, aud = get(f"/api/audio?id={res['id']}")
        check("audio(p_) 字节", st == 200 and aud[:4] == b"RIFF" and len(aud) > 10000,
              f"{len(aud)}B")

        # 5) remix：m4 推子 0 与 sep 推子 0.7
        keys = res["mixer"]["keys"]
        gains = {k: 1.0 for k in keys}
        gains["m4"] = 0.0
        gains["sep"] = 0.7
        t1 = time.time()
        st, m = post("/api/remix", {"job": jid, "gains": gains})
        dt = time.time() - t1
        check("POST /api/remix", m.get("ok") and m.get("url"), f"{dt:.2f}s {m}")
        st, aud2 = get(f"/api/audio?id={m['id']}")
        check("remix audio 字节", st == 200 and aud2[:4] == b"RIFF")
        check("  remix 快速", dt < 5.0, f"{dt:.2f}s")

        # 6) 即时 A/B 片段（/api/clips）：双侧、可播、内容与整轨一致
        st, c = post("/api/clips", {"id": res["id"], "orig_id": res["orig_id"],
                                    "start": 3.0, "dur": 2.0})
        check("POST /api/clips", c.get("ok") and c.get("a") and c.get("b"), str(c)[:120])
        check("  片段 3s→2.0s", c["dur"] == 2.0 and c["samples"] == 2 * 48000,
              f"dur={c['dur']} samples={c['samples']}")
        st, ca = get(c["a"]["url"])
        st2, cb = get(c["b"]["url"])
        check("  两侧片段均 RIFF 可播",
              ca[:4] == b"RIFF" and cb[:4] == b"RIFF" and len(ca) > 1000 and len(cb) > 1000,
              f"a={len(ca)}B b={len(cb)}B")
        check("  片段远小于整轨", len(ca) < len(aud) / 3, f"{len(ca)}B vs {len(aud)}B")
        # 内容一致性：片段应与整轨同区间逐样本一致（跳过 5ms 边缘淡化）
        sys.path.insert(0, ROOT)
        from livefix import wavio as _wavio

        full, _fsr = _wavio.read_wav(os.path.join(ROOT, "cache", f"{res['id']}.wav"))
        clip, _csr = _wavio.read_wav(os.path.join(ROOT, "cache", f"{c['a']['id']}.wav"))
        s0 = 3 * 48000
        ref = full[s0:s0 + clip.shape[0]]
        k = int(0.005 * 48000) + 2
        dmax = float(abs(clip[k:-k] - ref[k:-k]).max())
        check("  片段内容 == 整轨同区间", dmax < 1e-5, f"max|Δ|={dmax:.2e}")

        st, bad3 = post("/api/clips", {"id": res["id"], "start": 99999.0, "dur": 3.0})
        check("clips 越界 → ok:false", bad3.get("ok") is False, str(bad3)[:80])

        # 7) export：带 eq_bands + gains + job
        out_path = os.path.join(ROOT, "cache", "_e2e_final_export.wav")
        st, e = post("/api/export", {
            "id": None, "channels": "stereo", "bit_depth": 24, "loudnorm": True,
            "out_path": out_path, "job": jid, "gains": gains,
            "eq_bands": [{"f": 3000, "q": 1.0, "gain_db": 3.0},
                         {"f": 120, "q": 1.4, "gain_db": -3.0}]})
        check("export 烘焙(调音台+EQ)", e.get("ok") and os.path.exists(out_path),
              str({k: e.get(k) for k in ('ok', 'channels', 'duration', 'lufs')}))

        # 8) 拖拽导入（POST /api/upload：原始字节流 + ?name=）
        blob = open(SRC, "rb").read()
        st, up = post_raw("/api/upload?name=" + urllib.parse.quote("导入 测试.wav"), blob)
        check("POST /api/upload", st == 200 and up.get("ok"), str(up)[:120])
        check("  字节数一致", up.get("bytes") == len(blob), f"{up.get('bytes')} vs {len(blob)}")
        check("  落盘存在且大小一致",
              os.path.isfile(up.get("path", "")) and os.path.getsize(up["path"]) == len(blob),
              str(up.get("path")))
        check("  落在 cache/import/ 内",
              os.path.basename(os.path.dirname(up["path"])) == "import", str(up.get("dir")))
        check("  中文名与空格保留", up.get("name") == "导入 测试.wav", str(up.get("name")))
        # 导入的路径必须能直接喂给诊断（这就是拖拽导入的闭环）
        st, dg = post("/api/analyze", {"path": up["path"]})
        check("  导入物可直接诊断", dg.get("ok") and dg.get("sr"), str(dg)[:100])

        # 目录穿越 + 非法扩展名必须被拒，且不得落到 cache/import/ 之外
        st, trav = post_raw("/api/upload?name=" + urllib.parse.quote("../../evil.wav"), b"RIFF0000WAVE")
        check("  目录穿越被清洗到 import 内",
              trav.get("ok") and trav["name"] == "evil.wav"
              and os.path.basename(os.path.dirname(trav["path"])) == "import", str(trav)[:100])
        st, bad_up = post_raw("/api/upload?name=notes.txt", b"hello")
        check("  非音频扩展名 → ok:false", bad_up.get("ok") is False, str(bad_up)[:100])
        st, empty = post_raw("/api/upload?name=x.wav", b"")
        check("  空体 → ok:false", empty.get("ok") is False, str(empty)[:80])

        # 9) 错误路径
        st, bad = post("/api/remix", {"job": "job_nope", "gains": {}})
        check("remix 未知 job → ok:false", bad.get("ok") is False, str(bad))
        st, bad2 = get_json("/api/job?id=job_nope")
        check("job 未知 id → ok:false", bad2.get("ok") is False)

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

    fails = [n for n, ok in results if not ok]
    print("\n" + ("SERVER E2E: ALL PASS (%d)" % len(results) if not fails
                  else f"SERVER E2E: {len(fails)} FAILED -> {fails}"))
    sys.exit(0 if not fails else 1)


if __name__ == "__main__":
    main()
