"""platform.py — 平台探测与适配（triton/venv/锁/启动方式）。

仅使用标准库。logger 由调用方注入（鸭子类型），本模块不直接依赖 diag。
"""
from __future__ import annotations

import os
import sys
import json
import time
import shutil
import socket
import subprocess

import platform as _platform


# ---------------------------------------------------------------------------
# 平台 / 魔搭实例识别
# ---------------------------------------------------------------------------
def detect_platform() -> dict:
    system = _platform.system()  # Windows / Linux / Darwin
    is_ms = os.path.isdir("/mnt/workspace")
    return {"os": system, "arch": _platform.machine(), "is_modelscope": is_ms}


def is_modelscope() -> bool:
    return detect_platform()["is_modelscope"]


def check_system_torch(logger=None) -> bool:
    """检测系统 Python 是否已装 torch（魔搭镜像预装）。"""
    try:
        out = subprocess.run([sys.executable, "-c", "import torch; print(torch.__version__)"],
                             capture_output=True, text=True, timeout=30)
        ok = out.returncode == 0
        if logger:
            logger.info("P0", "torch", "系统 torch 检测", detail=(out.stdout.strip() if ok else out.stderr.strip()[:120]))
        return ok
    except Exception as e:  # noqa: BLE001
        if logger:
            logger.warn("P0", "torch", "系统 torch 检测失败", detail=str(e)[:120])
        return False


# ---------------------------------------------------------------------------
# GPU 探测
# ---------------------------------------------------------------------------
def detect_gpu() -> dict | None:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30).stdout.strip()
        if not out:
            return None
        line = out.splitlines()[0]
        name, mem, cap = [x.strip() for x in line.split(",")]
        vram_gb = float(mem.replace("MiB", "").replace("MB", "").strip()) / 1024.0
        return {"name": name, "vram_gb": round(vram_gb, 1), "compute_cap": cap}
    except Exception:  # noqa: BLE001
        return None


def compute_cap_ok(gpu: dict | None, min_cap: float = 8.0) -> bool:
    if not gpu:
        return False
    try:
        return float(gpu.get("compute_cap", "0")) >= min_cap
    except Exception:  # noqa: BLE001
        return False


# ---------------------------------------------------------------------------
# 网络探测（egress，§9.2）
# ---------------------------------------------------------------------------
def _probe_host(host: str, port: int, timeout: float = 5.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:  # noqa: BLE001
        return False


def network_probe(hosts: dict, logger=None) -> dict:
    """hosts: {name: url}；返回 {name: bool}。"""
    result = {}
    for name, url in hosts.items():
        try:
            host = url.split("://", 1)[1].split("/", 1)[0]
            port = 443 if url.startswith("https") else 80
        except Exception:  # noqa: BLE001
            result[name] = False
            continue
        ok = _probe_host(host, port)
        result[name] = ok
        if logger:
            logger.info("P0", "net", f"探测 {name}", detail=f"{url} -> {ok}")
    return result


DEFAULT_HOSTS = {
    "modelscope": "https://modelscope.cn",
    "hf-mirror": "https://hf-mirror.com",
    "huggingface": "https://huggingface.co",
    "github": "https://github.com",
    "gh-proxy": "https://gh-proxy.com",
    "gitclone": "https://gitclone.com",
}


# ---------------------------------------------------------------------------
# 并发锁（含 stale PID 释放，§6 规则6）
# ---------------------------------------------------------------------------
def _pid_alive(pid: int) -> bool:
    if os.name == "nt":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32  # type: ignore
            PROCESS_QUERY_INFORMATION = 0x400
            h = kernel32.OpenProcess(PROCESS_QUERY_INFORMATION, False, pid)
            if h:
                kernel32.CloseHandle(h)
                return True
            return False
        except Exception:  # noqa: BLE001
            return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


class FileLock:
    def __init__(self, path: str):
        self.path = path
        self.acquired = False

    def acquire(self, logger=None) -> bool:
        # stale 检测
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    info = json.load(f)
                if not _pid_alive(info.get("pid", -1)):
                    if logger:
                        logger.warn("P0", "lock", "检测到 stale 锁，自动释放",
                                    detail=str(info))
                    try:
                        os.remove(self.path)
                    except Exception:  # noqa: BLE001
                        pass
                else:
                    return False  # 持有者仍存活
            except Exception:  # noqa: BLE001
                try:
                    os.remove(self.path)
                except Exception:  # noqa: BLE401
                    pass
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump({"pid": os.getpid(), "ts": time.time()}, f)
            self.acquired = True
            return True
        except FileExistsError:
            return False

    def release(self):
        if self.acquired:
            try:
                if os.path.exists(self.path):
                    os.remove(self.path)
            except Exception:  # noqa: BLE001
                pass
            self.acquired = False


# ---------------------------------------------------------------------------
# venv 管理
# ---------------------------------------------------------------------------
def venv_python(root: str) -> str:
    if os.name == "nt":
        return os.path.join(root, "venv", "Scripts", "python.exe")
    return os.path.join(root, "venv", "bin", "python")


def ensure_venv(root: str, system_site_packages: bool, logger=None, ctx=None) -> str:
    """返回 venv 的 python 绝对路径；已存在且可用则复用，否则重建。

    v2.3.15：venv 创建健壮性。`python3 -m venv` 在部分魔搭/自定义镜像上会因
    ``ensurepip`` 不可用而退出码 1（典型：Debian/Ubuntu 缺 ``python3-venv``；或
    自定义 Python 构建未捆绑 ensurepip 引导轮）。此前 ``subprocess.run(check=True)``
    直接抛 ``CalledProcessError``，被顶层吞成无意义的 ``FATAL/unexpected``，且丢失
    真实 stderr。现改为依次尝试、捕获真实报错并优雅回退：

      1) ``python3 -m venv [--system-site-packages]``                    常规
      2) ``python3 -m venv [--system-site-packages] --without-pip``      绕过 ensurepip 引导
      3) 直接复用系统 Python（``sys.executable``）——本就 ``--system-site-packages``，
         依赖由镜像预置，避免污染/破坏系统环境（如镜像自带的 ROCm torch）

    任一成功即返回可用 python；全部失败也不抛错，最终回退到系统 Python 并置
    ``ctx.venv_fallback=True``（供 P2 跳过 requirements 安装）。
    """
    vpy = venv_python(root)
    if os.path.exists(vpy):
        try:
            r = subprocess.run([vpy, "-c", "import sys; print(sys.version)"],
                               capture_output=True, text=True, timeout=30)
            if r.returncode == 0:
                if logger:
                    logger.info("P2", "venv", "复用已有 venv")
                return vpy
        except Exception:  # noqa: BLE001
            pass
        if logger:
            logger.warn("P2", "venv", "venv 失效，重建", detail=vpy)
        shutil.rmtree(os.path.join(root, "venv"), ignore_errors=True)

    venv_dir = os.path.join(root, "venv")
    attempts = [
        ([], "标准 venv（--system-site-packages）"),
        (["--without-pip"], "venv --without-pip（绕过 ensurepip 引导）"),
    ]
    last_err = ""
    for extra, label in attempts:
        cmd = [sys.executable, "-m", "venv"]
        if system_site_packages:
            cmd.append("--system-site-packages")
        cmd.extend(extra)
        cmd.append(venv_dir)
        if logger:
            logger.info("P2", "venv", "创建 venv", detail=label)
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if r.returncode == 0 and os.path.exists(vpy):
            chk = subprocess.run([vpy, "-c", "import sys; print(sys.version)"],
                                 capture_output=True, text=True, timeout=30)
            if chk.returncode == 0:
                if logger:
                    logger.ok("P2", "venv", "venv 创建成功", detail=label)
                if ctx is not None:
                    # --without-pip 视为 fallback：pip 经 --system-site-packages 落到系统，
                    # P2 跳过 requirements 安装以免破坏系统环境
                    ctx.venv_fallback = bool(extra)
                return vpy
            last_err = (chk.stderr or "").strip()
        else:
            last_err = (r.stderr or r.stdout or "").strip()
        if logger:
            logger.warn("P2", "venv", "venv 创建失败",
                        detail={"label": label, "rc": r.returncode, "err": last_err[-300:]})

    # 全部 venv 方式失败 → 回退到系统 Python（永不抛错，优雅降级）
    if logger:
        logger.warn("P2", "venv", "venv 创建均失败（多为 ensurepip 不可用），回退到系统 Python 直接运行",
                    detail={"sys_executable": sys.executable, "last_err": last_err[-300:]})
    if ctx is not None:
        ctx.venv_fallback = True
    return sys.executable


# ---------------------------------------------------------------------------
# Windows triton 修复（F8，§8 P2）
# ---------------------------------------------------------------------------
def install_triton_windows(venv_python: str, logger=None) -> bool:
    try:
        subprocess.run([venv_python, "-m", "pip", "install", "triton-windows"],
                       check=True, capture_output=True, text=True, timeout=600)
        # 装后复验 torch 仍可用
        r = subprocess.run([venv_python, "-c", "import torch; print(torch.__version__)"],
                           capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            if logger:
                logger.warn("P2", "triton", "triton-windows 装后 torch 损坏，回滚",
                           detail=r.stderr.strip()[:200])
            subprocess.run([venv_python, "-m", "pip", "uninstall", "-y", "triton-windows"],
                           capture_output=True, text=True)
            return False
        if logger:
            logger.ok("P2", "triton", "triton-windows 安装并复验通过")
        return True
    except Exception as e:  # noqa: BLE001
        if logger:
            logger.warn("P2", "triton", "triton-windows 安装失败", detail=str(e)[:200])
        return False


# ---------------------------------------------------------------------------
# 磁盘可用空间
# ---------------------------------------------------------------------------
def disk_free(path: str) -> int:
    try:
        return shutil.disk_usage(path).free
    except Exception:  # noqa: BLE001
        return 0
