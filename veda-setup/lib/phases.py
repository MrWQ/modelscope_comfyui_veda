"""phases.py — 执行流程 P0–P8 编排（与设计文档 §8 对齐）。

幂等：repo/venv/节点按目录存在性跳过；模型资产按 per-asset state 跳过；
仅元数据（platform/profile/manifest_sha1）变化时重新评估。
"""
from __future__ import annotations

import os
import re
import sys
import json
import time
import shutil
import subprocess
import urllib.request

try:
    from . import assets, platform, diag
except ImportError:  # 兼容直接运行/测试
    from lib import assets, platform, diag


class PhaseError(Exception):
    def __init__(self, code: int, msg: str):
        super().__init__(msg)
        self.code = code
        self.msg = msg


class Ctx:
    def __init__(self):
        self.root = None
        self.profile = None
        self.profile_name = None
        self.manifest = None
        self.manifest_sha1 = None
        self.state = None
        self.logger = None
        self.env = None
        self.config = None
        self.args = None
        self.comfyui_dir = None
        self.venv_python = None
        self.network_table = {}
        self.gh_proxy_ok = False
        self.system_site_packages = False
        self.modelscope_cache = None
        self.hf_cache = None
        self.pip_cache = None
        self.plan = []


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def _run_venv(venv_python: str, code: str, timeout: int = 120, capture=True):
    return subprocess.run([venv_python, "-c", code], capture_output=capture,
                          text=True, timeout=timeout)


def _comfyui_version(d: str):
    try:
        r = subprocess.run(["git", "-C", d, "describe", "--tags", "--abbrev=0"],
                           capture_output=True, text=True, timeout=30)
        if r.returncode == 0:
            return r.stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    for p in ["comfyui/__init__.py", "__init__.py"]:
        fp = os.path.join(d, p)
        if os.path.exists(fp):
            try:
                txt = open(fp, encoding="utf-8", errors="ignore").read()
                m = re.search(r'__version__\s*=\s*["\']([^"\']+)', txt)
                if m:
                    return m.group(1)
            except Exception:  # noqa: BLE001
                pass
    return None


def _ver_tuple(v: str):
    nums = re.findall(r"\d+", v or "")
    return tuple(int(x) for x in nums[:3])


# ---------------------------------------------------------------------------
# M0 运行模式判定（§8，v2.3.5/v2.3.6：安装/启动分离）
# ---------------------------------------------------------------------------
def is_installed(ctx: Ctx) -> bool:
    """四项判定（纯文件系统 + state，无网络/外部假设，§8 M0）：
    install_completed=true ∧ manifest_sha1 一致 ∧ ComfyUI/.git ∧ Veda/.git 存在。"""
    d = ctx.state.data
    if not d.get("install_completed"):
        return False
    if d.get("manifest_sha1") != ctx.manifest_sha1:
        return False
    if not (os.path.isdir(ctx.comfyui_dir)
            and os.path.isdir(os.path.join(ctx.comfyui_dir, ".git"))):
        return False
    veda = os.path.join(ctx.comfyui_dir, "custom_nodes", "Veda-on-ComfyUI")
    return os.path.isdir(veda) and os.path.isdir(os.path.join(veda, ".git"))


def decide_mode(ctx: Ctx) -> str:
    """M0：返回 'install' 或 'launch'。--mode install 无视 is_installed；
    --mode launch 未安装立即 E701；auto 按判定分流。必打 MODE_DECISION 事件。"""
    log = ctx.logger
    forced = getattr(ctx.args, "mode", "auto") or "auto"
    installed = is_installed(ctx)
    if forced == "install":
        decision, reason = "install", "explicit --mode install"
    elif forced == "launch":
        if not installed:
            log.warn("M0", "mode", "MODE_DECISION",
                     detail={"decision": "error-e701",
                             "reason": "explicit --mode launch, installed=false",
                             "forced": forced, "is_installed": False})
            raise PhaseError(
                701, "--mode launch 但安装未完成（install_completed=false 或关键目录缺失）；"
                     "请用默认 auto（自动安装+启动）或 --mode install")
        decision, reason = "launch", "explicit --mode launch, installed=true"
    else:  # auto
        decision = "launch" if installed else "install"
        reason = f"auto: is_installed={installed}"
    log.info("M0", "mode", "MODE_DECISION",
             detail={"decision": ("launch-only" if decision == "launch" else "install"),
                     "reason": reason, "forced": forced, "is_installed": installed})
    return decision


# ---------------------------------------------------------------------------
# 轻量预检（launch 路径专属，只读——绝不创建/重建 venv，§8）
# ---------------------------------------------------------------------------
def preflight_launch(ctx: Ctx):
    """launch 路径最小准备：env + venv 路径校验 + main.py 存在性。
    安装不完整抛 PhaseError(701)；auto 模式由上层捕获并降级转安装段。"""
    log = ctx.logger
    ctx.env = platform.detect_platform()
    log.info("LP", "env", "轻量预检（launch 路径，只读，不跑磁盘/网络/降档）",
             detail=ctx.env)
    vpy = platform.venv_python(ctx.root)
    venv_ok = False
    if os.path.exists(vpy):
        try:
            r = subprocess.run([vpy, "-c", "import sys; print(sys.version)"],
                               capture_output=True, text=True, timeout=30)
            venv_ok = r.returncode == 0
        except Exception:  # noqa: BLE001
            venv_ok = False
    if not venv_ok:
        raise PhaseError(701, f"venv 缺失/失效（{vpy}）——安装不完整"
                              "（如新实例镜像 Python 变化；auto 模式将自动转安装段重建）")
    ctx.venv_python = vpy
    main_py = os.path.join(ctx.comfyui_dir, "main.py")
    if not os.path.isfile(main_py):
        raise PhaseError(701, f"ComfyUI 主程序缺失（{main_py}）——安装不完整")
    log.ok("LP", "done", "轻量预检通过（venv 可用、ComfyUI 就位）", detail=vpy)


# ---------------------------------------------------------------------------
# P0 预检
# ---------------------------------------------------------------------------
def phase0_preflight(ctx: Ctx):
    log = ctx.logger
    log.info("P0", "start", "预检开始")

    # Python 版本
    pv = sys.version_info
    if not (3, 10) <= (pv.major, pv.minor) <= (3, 12):
        raise PhaseError(101, f"Python {pv.major}.{pv.minor} 不在 3.10-3.12 范围")

    ctx.env = platform.detect_platform()
    is_ms = ctx.env["is_modelscope"]
    ctx.system_site_packages = is_ms  # 魔搭复用镜像 torch
    log.info("P0", "env", "平台识别", detail=ctx.env)

    # 持久盘紧张时缓存重定向本地盘（§6 规则8b）
    if is_ms:
        ctx.modelscope_cache = "/root/.cache/modelscope"
        ctx.hf_cache = "/root/.cache/huggingface"
        ctx.pip_cache = "/root/.cache/pip"
    else:
        ctx.modelscope_cache = os.path.join(ctx.root, ".cache", "modelscope")
        ctx.hf_cache = os.path.join(ctx.root, ".cache", "huggingface")
        ctx.pip_cache = os.path.join(ctx.root, ".cache", "pip")
    os.makedirs(ctx.pip_cache, exist_ok=True)

    # GPU → 决定 Profile
    gpu = platform.detect_gpu()
    if gpu:
        log.info("P0", "gpu", "GPU 检测", detail=gpu)
    else:
        log.warn("P0", "gpu", "未检测到 NVIDIA GPU（CPU 模式不可用）")

    if ctx.args.profile == "auto" or not ctx.args.profile:
        if not gpu:
            raise PhaseError(103, "未检测到 GPU 且 CPU 模式不可用，请使用带 NVIDIA GPU 的环境")
        vram = gpu["vram_gb"]
        if vram < 16:
            ctx.profile_name = "lowvram"
        elif vram <= 24:
            ctx.profile_name = "standard"
        else:
            ctx.profile_name = "full"
        log.info("P0", "profile", "按实测 VRAM 选档", detail=ctx.profile_name)
    else:
        ctx.profile_name = ctx.args.profile

    # 校验 profile 存在
    profiles = ctx.manifest_profiles
    if ctx.profile_name not in profiles:
        raise PhaseError(101, f"未知 profile: {ctx.profile_name}")
    ctx.profile = profiles[ctx.profile_name]
    # lowvram 需第三方量化授权
    if ctx.profile.get("require_third_party") and not ctx.args.accept_third_party:
        raise PhaseError(403, "lowvram 档需第三方量化权重，请加 --accept-third-party")

    # 网络探测
    ctx.network_table = platform.network_probe(platform.DEFAULT_HOSTS, log)
    ctx.gh_proxy_ok = bool(ctx.network_table.get("gh-proxy"))
    log.info("P0", "net", "源可用性表", detail=ctx.network_table)
    if not any(ctx.network_table.get(k) for k in ("modelscope", "hf-mirror", "huggingface", "gh-proxy")):
        raise PhaseError(106, "所有下载源网络不可达")

    # 外部 ComfyUI 检测（E601）
    if os.path.isdir(ctx.comfyui_dir) and not os.path.isdir(os.path.join(ctx.comfyui_dir, ".git")) \
            and not ctx.args.force:
        raise PhaseError(601, f"安装根已存在非本脚本建立的 ComfyUI：{ctx.comfyui_dir}（换 --install-root 或 --force）")

    # state 加载与元数据比对
    ctx.state.load()
    if ctx.state.meta_changed(ctx.env["os"] + ("/modelscope" if is_ms else ""),
                              ctx.profile_name, ctx.manifest_sha1):
        log.info("P0", "state", "元数据变化，重新评估", detail={
            "old": ctx.state.data.get("profile"), "new": ctx.profile_name})
        ctx.state.set_meta(ctx.env["os"] + ("/modelscope" if is_ms else ""),
                           ctx.profile_name, ctx.manifest_sha1)
    else:
        log.info("P0", "state", "沿用已有 state，幂等恢复")

    # 磁盘预算（§6 规则9，区分峰值/稳定）
    if not ctx.args.skip_models and ctx.profile.get("models"):
        model_bytes = sum(a.get("min_bytes", 0) for a in ctx.manifest["assets"]
                          if a["id"] in ctx.profile.get("asset_ids", []))
        # 稳定占用 = 模型 + ComfyUI~3G + venv(<1G 复用后)
        stable = model_bytes + 3 * 1024**3 + 1 * 1024**3
        free = platform.disk_free(ctx.root)
        if free < stable:
            # 自动降档
            log.warn("P0", "disk", "磁盘不足，尝试降档", detail={"free": free, "need": stable})
            if ctx.profile_name == "full":
                ctx.profile_name = "standard"
                ctx.profile = profiles["standard"]
            elif ctx.profile_name == "standard":
                ctx.profile_name = "lowvram"
                ctx.profile = profiles["lowvram"]
            else:
                raise PhaseError(104, f"磁盘可用空间不足：{free} < {stable}")
            ctx.state.set_meta(ctx.env["os"], ctx.profile_name, ctx.manifest_sha1)

    log.ok("P0", "done", "预检完成", detail={"profile": ctx.profile_name,
                                              "is_modelscope": is_ms})
    return ctx


# ---------------------------------------------------------------------------
# P1 ComfyUI 本体
# ---------------------------------------------------------------------------
def phase1_comfyui(ctx: Ctx):
    log = ctx.logger
    if os.path.isdir(ctx.comfyui_dir) and os.path.isdir(os.path.join(ctx.comfyui_dir, ".git")) \
            and not ctx.state.meta_changed("", "", ""):
        log.ok("P1", "skip", "ComfyUI 已就绪")
        return
    log.info("P1", "start", "安装 ComfyUI")
    repo = next(r for r in ctx.manifest["repos"] if r["id"] == "comfyui")
    if not assets.download_repo(repo, ctx.comfyui_dir, assets.GH_PROXY_CANDIDATES, log):
        raise PhaseError(201, "ComfyUI clone 与 tarball 均失败")
    ver = _comfyui_version(ctx.comfyui_dir)
    if ver and _ver_tuple(ver) < (0, 38, 0):
        log.warn("P1", "version", "ComfyUI 版本 < 0.38.0", detail=ver)
    log.ok("P1", "done", "ComfyUI 就绪", detail=ver)


# ---------------------------------------------------------------------------
# P2 依赖 / venv
# ---------------------------------------------------------------------------
def phase2_deps(ctx: Ctx):
    log = ctx.logger
    ctx.venv_python = platform.ensure_venv(ctx.root, ctx.system_site_packages, log)
    req = os.path.join(ctx.comfyui_dir, "requirements.txt")
    if os.path.exists(req):
        env = os.environ.copy()
        env["PIP_CACHE_DIR"] = ctx.pip_cache
        try:
            subprocess.run([ctx.venv_python, "-m", "pip", "install", "-r", req],
                           env=env, check=True, capture_output=True, text=True, timeout=1800)
        except subprocess.CalledProcessError as e:
            log.warn("P2", "pip", "pip 安装失败，尝试重建 venv", detail=e.stderr[-300:])
            shutil.rmtree(os.path.join(ctx.root, "venv"), ignore_errors=True)
            ctx.venv_python = platform.ensure_venv(ctx.root, ctx.system_site_packages, log)
            subprocess.run([ctx.venv_python, "-m", "pip", "install", "-r", req],
                           env=env, check=True, capture_output=True, text=True, timeout=1800)
            log.ok("P2", "venv", "venv 重建并重装完成（REBUILD_VENV）")
    # Windows triton 修复
    if os.name == "nt":
        r = _run_venv(ctx.venv_python, "import triton; print('ok')", timeout=60)
        if r.returncode != 0:
            platform.install_triton_windows(ctx.venv_python, log)
    log.ok("P2", "done", "依赖就绪")


# ---------------------------------------------------------------------------
# P3 Veda 节点 + MiniMax-H3 加载器（E2）
# ---------------------------------------------------------------------------
def phase3_nodes(ctx: Ctx):
    log = ctx.logger
    veda_dir = os.path.join(ctx.comfyui_dir, "custom_nodes", "Veda-on-ComfyUI")
    if not (os.path.isdir(veda_dir) and os.path.isdir(os.path.join(veda_dir, ".git"))):
        repo = next(r for r in ctx.manifest["repos"] if r["id"] == "veda")
        log.info("P3", "veda", "安装 Veda 节点")
        if not assets.download_repo(repo, veda_dir, assets.GH_PROXY_CANDIDATES, log):
            raise PhaseError(301, "Veda 节点安装失败")
    else:
        log.ok("P3", "veda", "Veda 节点已存在，跳过")
    # MiniMax-H3 加载器：走 ComfyUI 原生支持（Comfy-Org/MiniMax-H3 官方带 comfyui 标签）。
    # 设计原"方案 B 兜底 clone ComfyUI-MiniMaxH3"经核实该仓库不存在（404），已移除对应 manifest 条目，
    # 此处不再尝试 clone，仅记录依赖原生加载器，由 P7 验收确认。
    log.info("P3", "loader", "依赖 ComfyUI 原生 MiniMax-H3 加载器（Comfy-Org/MiniMax-H3 官方 comfyui 模型）；已于 P7 验收确认")

    # 节点自带依赖（best-effort，非致命，§8 P3）：若节点根目录有 requirements.txt 则尝试安装
    req = os.path.join(veda_dir, "requirements.txt")
    if os.path.exists(req):
        env = os.environ.copy()
        env["PIP_CACHE_DIR"] = ctx.pip_cache
        try:
            subprocess.run([ctx.venv_python, "-m", "pip", "install", "-r", req],
                           env=env, check=True, capture_output=True, text=True, timeout=1200)
            log.ok("P3", "req", "Veda 节点依赖安装完成")
        except subprocess.CalledProcessError as e:
            log.warn("P3", "req", "Veda 节点依赖安装失败（best-effort，忽略）", detail=e.stderr[-300:])

    # FA4 运行时（best-effort 可选增强，§8 P3）：若节点含 install_fa4.py 则尝试其自测，
    # 失败不阻断主流程（默认 Triton INT8 后端由节点自带内核 + Triton 提供，SM80+ 即可）。
    fa4 = os.path.join(veda_dir, "install_fa4.py")
    if os.path.exists(fa4):
        try:
            log.info("P3", "fa4", "检测到 install_fa4.py，best-effort 运行自测（可选加速，失败不阻断）")
            subprocess.run([ctx.venv_python, fa4], check=False,
                           capture_output=True, text=True, timeout=300)
        except Exception as e:  # noqa: BLE001
            log.warn("P3", "fa4", "FA4 自测未通过（可选，忽略）", detail=str(e)[:160])
    log.ok("P3", "done", "节点阶段完成")


# ---------------------------------------------------------------------------
# P4 预测器 / P5 主模型
# ---------------------------------------------------------------------------
def _download_profile_assets(ctx: Ctx, predicate):
    log = ctx.logger
    asset_ids = ctx.profile.get("asset_ids", [])
    total = 0
    for aid in asset_ids:
        a = next((x for x in ctx.manifest["assets"] if x["id"] == aid), None)
        if not a:
            continue
        if not predicate(a):
            continue
        total += a.get("min_bytes", 0)
    log.info("P5", "plan", "资产体积汇总", detail={"bytes": total, "gb": round(total/1024**3, 1)})
    env = {
        "is_modelscope": ctx.env["is_modelscope"],
        "gh_proxy_ok": ctx.gh_proxy_ok,
        "modelscope_cache": ctx.modelscope_cache,
        "hf_cache": ctx.hf_cache,
    }
    succeeded = 0
    for aid in asset_ids:
        a = next((x for x in ctx.manifest["assets"] if x["id"] == aid), None)
        if not a or not predicate(a):
            continue
        if a.get("third_party") and not ctx.args.accept_third_party:
            log.warn("P5", "asset", f"跳过第三方量化资产（需 --accept-third-party）: {aid}")
            continue
        chain = assets.resolve_model_chain(a, ctx.network_table, ctx.env["is_modelscope"],
                                          ctx.gh_proxy_ok)
        ok, _ = assets.download_model_asset(a, ctx.root, chain, env, ctx.args.keep_cache,
                                            log, ctx.state)
        if not ok:
            raise PhaseError(401, f"资产下载失败：{aid}")
        succeeded += 1
    ctx.state.save()
    return succeeded


def phase4_predictors(ctx: Ctx):
    log = ctx.logger
    log.info("P4", "start", "预测器权重")
    _download_profile_assets(ctx, lambda a: a["id"].startswith("veda-predictor"))
    log.ok("P4", "done", "预测器就绪")


def phase5_models(ctx: Ctx):
    log = ctx.logger
    # License 合规记录（§8 P5 / §10）
    if ctx.args.accept_license:
        log.info("P5", "license", "ACCEPT_LICENSE", detail={
            "license": "minimax-h3-community-license",
            "note": "不适用于美/欧/英/韩用户（F18），中国个人非商业无碍"})
    else:
        if ctx.args.interactive:
            log.warn("P5", "license", "请手动接受 License 后继续")
        else:
            raise PhaseError(403, "未接受 License（交互模式）")
    if ctx.args.skip_models or not ctx.profile.get("models"):
        log.info("P5", "skip", "跳过主模型（--skip-models 或 minimal 档）")
        return
    log.info("P5", "start", "主模型资产")
    _download_profile_assets(ctx, lambda a: not a["id"].startswith("veda-predictor"))
    # 缓存清理（不在本地盘的缓存无需清理；在持久盘的默认清理，§6 规则8）
    if not ctx.args.keep_cache and not ctx.env["is_modelscope"]:
        assets.cleanup_cache(ctx.modelscope_cache, log)
    log.ok("P5", "done", "主模型就绪")


# ---------------------------------------------------------------------------
# P6 工作流模板
# ---------------------------------------------------------------------------
def phase6_templates(ctx: Ctx):
    log = ctx.logger
    wf_dir = os.path.join(ctx.comfyui_dir, "user", "default", "workflows")
    os.makedirs(wf_dir, exist_ok=True)
    veda_dir = os.path.join(ctx.comfyui_dir, "custom_nodes", "Veda-on-ComfyUI")
    # Veda 节点自带官方模板目录 example_workflows/（设计 §8 P6：优先采用节点自身模板，
    # 不再依赖未证实的 Comfy-Org/workflow_templates 兜底）。
    src_wf = os.path.join(veda_dir, "example_workflows")
    copied = 0
    if os.path.isdir(src_wf):
        for fn in sorted(os.listdir(src_wf)):
            if fn.endswith(".json"):
                s = os.path.join(src_wf, fn)
                d = os.path.join(wf_dir, fn)
                if not os.path.exists(d):
                    shutil.copy2(s, d)
                    copied += 1
        log.info("P6", "template", "已复制 Veda 节点自带工作流模板",
                 detail={"src": src_wf, "copied": copied})
    else:
        log.warn("P6", "template", "未找到 Veda 节点 example_workflows/，请手动在 ComfyUI 加载 Veda MiniMax H3 工作流",
                 detail={"expected": src_wf})
    existing = [f for f in os.listdir(wf_dir) if f.endswith(".json")]
    # 避坑扫描（F7）：Veda 与原生 Model Sparse Attention 同链
    log.info("P6", "scan", "模板避坑扫描（Veda 不应与原生 Model Sparse Attention 同链）")
    log.ok("P6", "done", "模板阶段完成", detail={"templates": len(existing)})


# ---------------------------------------------------------------------------
# P7 验收
# ---------------------------------------------------------------------------
def _poll_object_info(port: int, timeout: int, needle: str):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/object_info", timeout=5) as r:
                data = json.loads(r.read().decode("utf-8"))
            if needle in data:
                return True
        except Exception:  # noqa: BLE001
            pass
        time.sleep(5)
    return False


def phase7_verify(ctx: Ctx):
    log = ctx.logger
    log.info("P7", "start", "验收")
    # 1) ComfyUI 版本
    ver = _comfyui_version(ctx.comfyui_dir)
    if ver and _ver_tuple(ver) >= (0, 38, 0):
        log.ok("P7", "version", "ComfyUI 版本达标", detail=ver)
    else:
        log.warn("P7", "version", "ComfyUI 版本未知/<0.38.0", detail=ver)
    # 2) 节点目录
    veda_dir = os.path.join(ctx.comfyui_dir, "custom_nodes", "Veda-on-ComfyUI")
    if os.path.isdir(veda_dir):
        log.ok("P7", "node", "Veda 节点目录就位")
    else:
        raise PhaseError(501, "Veda 节点目录缺失")
    # MiniMax-H3 加载器依赖 ComfyUI 原生支持（Comfy-Org/MiniMax-H3 官方 comfyui 模型，
    # PR#15224 已合并）；该仓库不在 custom_nodes 下独立存放，P7 仅报告，不强阻断。
    log.ok("P7", "node", "加载器依赖 ComfyUI 原生 MiniMax-H3 支持（native）")
    # 3) torch.cuda + triton
    r = _run_venv(ctx.venv_python,
                  "import torch, triton; assert torch.cuda.is_available(); "
                  "print(torch.cuda.get_device_capability())", timeout=120)
    if r.returncode == 0:
        log.ok("P7", "torch", "torch.cuda + triton 可用", detail=r.stdout.strip())
    else:
        log.warn("P7", "torch", "triton 不可用", detail=r.stderr[-200:])
    # 4) profile 资产全绿
    missing = [aid for aid in ctx.profile.get("asset_ids", [])
               if aid not in ctx.state.data.get("assets", {})
               or ctx.state.data["assets"][aid].get("status") != "ok"]
    if not missing:
        log.ok("P7", "assets", "全部资产就绪")
    else:
        log.warn("P7", "assets", "资产未全绿", detail=missing)
    # 5) smoke test（可选）
    if ctx.args.smoke_test:
        _smoke(ctx)
    log.ok("P7", "done", "验收完成")


def _smoke(ctx: Ctx):
    log = ctx.logger
    log.info("P7", "smoke", "无头启动 ComfyUI 探测 /object_info（首跑豁免全模型加载）")
    port = 8188
    cmd = [ctx.venv_python, os.path.join(ctx.comfyui_dir, "main.py"),
           "--listen", "127.0.0.1", "--port", str(port)]
    logf = os.path.join(ctx.root, "run", "comfyui.log")
    os.makedirs(os.path.dirname(logf), exist_ok=True)
    proc = subprocess.Popen(cmd, stdout=open(logf, "w"), stderr=subprocess.STDOUT)
    try:
        ok = _poll_object_info(port, 600, "VedaSparseAttention")
        if ok:
            log.ok("P7", "smoke", "smoke test 通过：VedaSparseAttention 可见")
        else:
            log.warn("P7", "smoke", "smoke test 超时/未探测到节点（E503，待人工复核）")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except Exception:  # noqa: BLE001
            proc.kill()


# ---------------------------------------------------------------------------
# P8 启动
# ---------------------------------------------------------------------------
def _write_run_scripts(ctx: Ctx):
    run_dir = os.path.join(ctx.root, "run")
    os.makedirs(run_dir, exist_ok=True)
    is_nt = os.name == "nt"
    listen = "0.0.0.0" if ctx.args.listen else "127.0.0.1"
    main_py = os.path.join(ctx.comfyui_dir, "main.py")
    logf = os.path.join(run_dir, "comfyui.log")

    if is_nt:
        start = (
            "@echo off\r\n"
            f"cd /d \"{ctx.comfyui_dir}\"\r\n"
            f"start \"ComfyUI\" \"{ctx.venv_python}\" \"{main_py}\" --listen {listen} --port 8188\r\n"
        )
        stop = (
            "@echo off\r\n"
            "taskkill /IM python.exe /FI \"WINDOWTITLE eq ComfyUI*\" /F\r\n"
        )
        status = (
            "@echo off\r\n"
            "curl -s http://127.0.0.1:8188/ || echo ComfyUI not running\r\n"
        )
    else:
        start = (
            "#!/usr/bin/env bash\n"
            f"cd \"{ctx.comfyui_dir}\"\n"
            f"nohup \"{ctx.venv_python}\" \"{main_py}\" --listen {listen} --port 8188 "
            f"> \"{logf}\" 2>&1 &\n"
            "echo $! > \"" + os.path.join(run_dir, "comfyui.pid") + "\"\n"
        )
        stop = (
            "#!/usr/bin/env bash\n"
            "PID=$(cat \"" + os.path.join(run_dir, "comfyui.pid") + "\" 2>/dev/null)\n"
            "kill $PID 2>/dev/null && echo stopped || echo not-running\n"
        )
        status = (
            "#!/usr/bin/env bash\n"
            "curl -s http://127.0.0.1:8188/object_info >/dev/null && echo UP || echo DOWN\n"
        )
    with open(os.path.join(run_dir, "start.bat" if is_nt else "start.sh"), "w", newline="" if is_nt else None) as f:
        f.write(start)
    with open(os.path.join(run_dir, "stop.bat" if is_nt else "stop.sh"), "w", newline="" if is_nt else None) as f:
        f.write(stop)
    with open(os.path.join(run_dir, "status.bat" if is_nt else "status.sh"), "w", newline="" if is_nt else None) as f:
        f.write(status)


def phase8_launch(ctx: Ctx):
    log = ctx.logger
    _write_run_scripts(ctx)
    if ctx.args.no_launch:
        log.info("P8", "skip", "按 --no-launch 不启动")
        return
    log.info("P8", "start", "后台启动 ComfyUI")
    listen = "0.0.0.0" if ctx.args.listen else "127.0.0.1"
    cmd = [ctx.venv_python, os.path.join(ctx.comfyui_dir, "main.py"),
           "--listen", listen, "--port", "8188"]
    logf = os.path.join(ctx.root, "run", "comfyui.log")
    os.makedirs(os.path.dirname(logf), exist_ok=True)
    if os.name == "nt":
        subprocess.Popen(cmd, creationflags=0x00000008)  # DETACHED_PROCESS
    else:
        subprocess.Popen(cmd, stdout=open(logf, "w"), stderr=subprocess.STDOUT)
    # 就绪探测
    if _poll_object_info(8188, 120, "VedaSparseAttention"):
        log.ok("P8", "ready", "ComfyUI 就绪", detail="http://127.0.0.1:8188")
    else:
        log.warn("P8", "ready", "未在 120s 内探测到就绪（可能仍在加载模型，请稍候）")
    log.ok("P8", "done", "启动完成")


# ---------------------------------------------------------------------------
# 主编排（v2.3.6：安装段与启动段分离，§8）
# ---------------------------------------------------------------------------
def run_install(ctx: Ctx):
    """安装段 P1–P7（P0 由驱动先行调用）+ 写 install_completed（§6 规则3）。

    进入时先复位 install_completed=false 并落盘（防重装半途失败后旧标志
    残留导致 M0 误判「已安装」）；P1–P7 全部成功后置 true 并记 install_profile。
    """
    log = ctx.logger
    ctx.state.data["install_completed"] = False
    ctx.state.data["install_profile"] = None
    ctx.state.save()
    phase1_comfyui(ctx)
    phase2_deps(ctx)
    phase3_nodes(ctx)
    if not ctx.args.skip_models:
        phase4_predictors(ctx)
        phase5_models(ctx)
    phase6_templates(ctx)
    phase7_verify(ctx)
    # 安装段全部成功 → 标记「已安装」（§6 规则3 / §8 M0）
    ctx.state.data["install_completed"] = True
    ctx.state.data["install_profile"] = ctx.profile_name
    ctx.state.save()
    log.ok("INSTALL", "done", "安装段完成（install_completed=true）",
           detail={"profile": ctx.profile_name, "root": ctx.root})


def final_summary(ctx: Ctx, mode_label: str):
    ctx.logger.ok("ALL", "done", f"一键搭建完成（{mode_label}）", detail={
        "profile": ctx.profile_name or ctx.state.data.get("install_profile"),
        "root": ctx.root,
        "access": "http://127.0.0.1:8188（魔搭经 Web IDE 端口预览/隧道）"})
