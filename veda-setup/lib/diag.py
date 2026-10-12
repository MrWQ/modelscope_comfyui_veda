"""diag.py — 结构化日志、错误码注册表、幂等状态读写、Doctor 诊断包。

严格仅使用标准库，保证在全新实例上开箱即用。
"""
from __future__ import annotations

import json
import os
import sys
import time
import shutil
import socket
import subprocess
import tempfile
import tarfile
import zipfile

import platform as _platform

# ---------------------------------------------------------------------------
# 错误码注册表（与设计文档 §11.2 对齐）
# ---------------------------------------------------------------------------
ERROR_CODES = {
    0:   ("SUCCESS", "成功"),
    101: ("E101", "操作系统不受支持"),
    102: ("E102", "GPU 计算能力 < SM80 或非 NVIDIA"),
    103: ("E103", "未检测到 GPU（CPU 模式不可用）"),
    104: ("E104", "磁盘可用空间不足"),
    105: ("E105", "安装根不可写"),
    106: ("E106", "所有下载源网络全不可达"),
    201: ("E201", "ComfyUI clone 与 tarball 均失败"),
    202: ("E202", "ComfyUI 版本 < 0.38.0"),
    203: ("E203", "pip 安装失败"),
    204: ("E204", "venv 重建失败"),
    301: ("E301", "Veda 节点（及兜底）安装失败"),
    401: ("E401", "资产所有源下载失败"),
    402: ("E402", "资产校验失败"),
    403: ("E403", "未接受 License（交互模式）"),
    501: ("E501", "验收：版本/资产缺失"),
    502: ("E502", "验收：triton 不可用"),
    503: ("E503", "验收：object_info 无 Veda/加载器节点"),
    601: ("E601", "安装根存在外来 ComfyUI"),
    602: ("E602", "安装根无写权限"),
    603: ("E603", "并发锁冲突（另一安装进行中）"),
    701: ("E701", "运行模式：--mode launch 但安装不完整"),
}


def error_name(code: int) -> str:
    return ERROR_CODES.get(code, ("E_UNKNOWN", "未知错误"))[0]


# ---------------------------------------------------------------------------
# 辅助：执行命令并取 stdout（诊断用，不抛异常）
# ---------------------------------------------------------------------------
def _run(cmd: str, timeout: int = 30) -> str:
    try:
        out = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                             errors="replace", timeout=timeout)
        return (out.stdout or "") + (out.stderr or "")
    except Exception as e:  # noqa: BLE001
        return f"<error: {e}>"


# ---------------------------------------------------------------------------
# 结构化日志
# ---------------------------------------------------------------------------
class Logger:
    COLORS = {"INFO": "", "WARN": "\033[33m", "ERROR": "\033[31m", "OK": "\033[32m"}

    def __init__(self, log_dir: str, run_id: str | None = None):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        if run_id is None:
            run_id = time.strftime("%Y%m%d_%H%M%S")
        self.run_id = run_id
        self.jsonl_path = os.path.join(log_dir, f"setup_{run_id}.jsonl")
        self._fh = open(self.jsonl_path, "a", encoding="utf-8")

    def _emit(self, level, phase, step, event, detail=None, duration_ms=None, error_code=None):
        rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "level": level,
               "phase": phase, "step": step, "event": event}
        if detail is not None:
            rec["detail"] = detail
        if duration_ms is not None:
            rec["duration_ms"] = duration_ms
        if error_code is not None:
            rec["error_code"] = error_code
        line = json.dumps(rec, ensure_ascii=False)
        try:
            self._fh.write(line + "\n")
            self._fh.flush()
        except Exception:  # noqa: BLE001
            pass
        c = self.COLORS.get(level, "")
        reset = "\033[0m" if c else ""
        msg = f"[{level}] {phase}/{step} {event}"
        if detail:
            if isinstance(detail, dict):
                detail = json.dumps(detail, ensure_ascii=False)
            msg += f" | {detail}"
        print(f"{c}{msg}{reset}", flush=True)

    def info(self, phase, step, event, **kw):
        self._emit("INFO", phase, step, event, **kw)

    def ok(self, phase, step, event, **kw):
        self._emit("OK", phase, step, event, **kw)

    def warn(self, phase, step, event, **kw):
        self._emit("WARN", phase, step, event, **kw)

    def error(self, phase, step, event, **kw):
        self._emit("ERROR", phase, step, event, **kw)

    def close(self):
        try:
            self._fh.close()
        except Exception:  # noqa: BLE001
            pass

    def rotate(self, keep_jsonl: int = 10, keep_diag: int = 5):
        """仅清理脚本自有日志/诊断包（不触碰其他目录，见 §12）。"""
        try:
            files = [f for f in os.listdir(self.log_dir)
                     if f.startswith("setup_") or f.startswith("diagnose_")]
            jsonls = sorted([f for f in files if f.startswith("setup_")])
            diags = sorted([f for f in files if f.startswith("diagnose_")])
            for old in jsonls[:-keep_jsonl]:
                os.remove(os.path.join(self.log_dir, old))
            for old in diags[:-keep_diag]:
                os.remove(os.path.join(self.log_dir, old))
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# 幂等状态读写（跨实例恢复依据，§6 规则3/5）
# ---------------------------------------------------------------------------
class State:
    DEFAULT = {"platform": None, "profile": None, "manifest_sha1": None,
               "assets": {}, "install_completed": False, "install_profile": None}

    def __init__(self, root: str):
        self.path = os.path.join(root, ".setup-state.json")
        self.data = dict(self.DEFAULT)

    def load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
            except Exception:  # noqa: BLE001
                bak = f"{self.path}.corrupt-{int(time.time())}"
                try:
                    shutil.move(self.path, bak)
                except Exception:  # noqa: BLE001
                    pass
                self.data = dict(self.DEFAULT)
        return self.data

    def save(self):
        tmp = f"{self.path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.path)

    def set_meta(self, platform, profile, manifest_sha1):
        self.data["platform"] = platform
        self.data["profile"] = profile
        self.data["manifest_sha1"] = manifest_sha1

    def meta_changed(self, platform, profile, manifest_sha1) -> bool:
        return (self.data.get("platform") != platform
                or self.data.get("profile") != profile
                or self.data.get("manifest_sha1") != manifest_sha1)

    def is_asset_done(self, asset_id: str, fingerprint: str) -> bool:
        a = self.data.get("assets", {}).get(asset_id)
        return bool(a and a.get("status") == "ok" and a.get("fingerprint") == fingerprint)

    def record_asset(self, asset_id, status, size=None, fingerprint=None, source=None):
        self.data.setdefault("assets", {})[asset_id] = {
            "status": status, "size": size, "fingerprint": fingerprint,
            "source": source, "ts": time.strftime("%Y-%m-%dT%H:%M%S"),
        }


# ---------------------------------------------------------------------------
# Doctor 诊断包
# ---------------------------------------------------------------------------
def _recent_log(log_dir: str, n: int = 2000) -> str:
    try:
        jsonls = sorted([f for f in os.listdir(log_dir) if f.startswith("setup_")],
                        reverse=True)
        if not jsonls:
            return "(no log)"
        with open(os.path.join(log_dir, jsonls[0]), "r", encoding="utf-8") as f:
            lines = f.readlines()[-n:]
        return "".join(lines)
    except Exception as e:  # noqa: BLE001
        return f"(error reading log: {e})"


def _file_tree(root: str, max_files: int = 2000) -> str:
    out = []
    count = 0
    for dirpath, dirnames, filenames in os.walk(root):
        # 跳过缓存与 venv 细节（仅列顶层结构，控制体积）
        if any(seg in dirpath for seg in ("/.git", "/.cache", "/venv")):
            if "/.cache" in dirpath or "/venv" in dirpath:
                out.append(f"{dirpath}/  (skipped)")
                dirnames[:] = []
                continue
        depth = dirpath.replace(root, "").count(os.sep)
        if depth > 3:
            continue
        for fn in filenames:
            if count >= max_files:
                out.append("... (truncated)")
                return "\n".join(out)
            fp = os.path.join(dirpath, fn)
            try:
                sz = os.path.getsize(fp)
            except Exception:  # noqa: BLE001
                sz = -1
            out.append(f"{dirpath}/  {fn}  ({sz} bytes)")
            count += 1
    return "\n".join(out)


def diagnose(root: str, log_dir: str, state_data: dict, config: dict,
             network_table: dict, version_snapshot: dict) -> str:
    """收集环境快照+配置+state+日志+文件树，打包 tar.gz(非Win)/zip(Win)。"""
    td = tempfile.mkdtemp(prefix="veda_diag_")
    try:
        snap = []
        snap.append("uname: " + " ".join(_platform.uname()))
        snap.append("python: " + sys.version.replace("\n", " "))
        snap.append("nvidia-smi:\n" + (_run("nvidia-smi") or "n/a"))
        snap.append("pip freeze:\n" + (_run(f"{sys.executable} -m pip freeze") or "n/a"))
        snap.append("df -h:\n" + (_run("df -h") or _run("wmic logicaldisk get size,freespace,caption") or "n/a"))
        snap.append("version snapshot:\n" + json.dumps(version_snapshot, ensure_ascii=False, indent=2))
        with open(os.path.join(td, "env_snapshot.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(snap))

        with open(os.path.join(td, "network_probe.json"), "w", encoding="utf-8") as f:
            json.dump(network_table, f, ensure_ascii=False, indent=2)
        with open(os.path.join(td, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        with open(os.path.join(td, "setup-state.json"), "w", encoding="utf-8") as f:
            json.dump(state_data, f, ensure_ascii=False, indent=2)
        with open(os.path.join(td, "recent_log.txt"), "w", encoding="utf-8") as f:
            f.write(_recent_log(log_dir))
        with open(os.path.join(td, "file_tree.txt"), "w", encoding="utf-8") as f:
            f.write(_file_tree(root))

        run_log = os.path.join(root, "run", "comfyui.log")
        if os.path.exists(run_log):
            with open(run_log, "r", encoding="utf-8", errors="replace") as f:
                tail = "".join(f.readlines()[-500:])
            with open(os.path.join(td, "comfyui_run_log_tail.txt"), "w", encoding="utf-8") as f:
                f.write(tail)

        ts = time.strftime("%Y%m%d_%H%M%S")
        if os.name == "nt":
            arc = os.path.join(log_dir, f"diagnose_{ts}.zip")
            with zipfile.ZipFile(arc, "w", zipfile.ZIP_DEFLATED) as z:
                for fn in os.listdir(td):
                    z.write(os.path.join(td, fn), fn)
        else:
            arc = os.path.join(log_dir, f"diagnose_{ts}.tar.gz")
            with tarfile.open(arc, "w:gz") as t:
                for fn in os.listdir(td):
                    t.add(os.path.join(td, fn), fn)
        return arc
    finally:
        shutil.rmtree(td, ignore_errors=True)
