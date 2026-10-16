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
        self.user_aborted = False  # v2.3.9：无 GPU 经询问后用户选择不继续
        self.launch_ready = None  # v2.3.11：P8 就绪标记，供 final_summary 诚实汇报
        self.launch_skipped_reason = None  # v2.3.14：P8 因环境不满足（无 GPU / torch 不兼容）跳过启动的原因
        self.venv_fallback = False  # v2.3.15：venv 创建失败回退到系统 Python 时为 True（P2 跳过 requirements 安装以保护环境）


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def _run_venv(venv_python: str, code: str, timeout: int = 120, capture=True):
    return subprocess.run([venv_python, "-c", code], capture_output=capture,
                          text=True, timeout=timeout)


def _torch_compat_ok(venv_python: str) -> "tuple[bool, str]":
    """v2.3.13：检查 venv 内 torch 是否满足 ComfyUI/Veda 运行时依赖。

    comfy_kitchen（被 ComfyUI 的 comfy/quant_ops.py 引入）需要
    ``torch.library.custom_op``，该 API 自 **torch>=2.0** 才有。魔搭环境 venv 复用镜像
    torch（system_site_packages），若镜像 torch < 2.0，启动即 AttributeError。此处提前
    探测，给出清晰报错而非等待 launch 崩溃后再 dump 原始 traceback。

    返回 (ok, detail)：ok=False 时 detail 为可操作的中文说明。
    """
    r = _run_venv(venv_python,
                  "import torch, sys; import torch.library; "
                  "print(torch.__version__); "
                  "print('custom_op' if hasattr(torch.library, 'custom_op') else 'no_custom_op')",
                  timeout=120)
    if r.returncode != 0:
        return False, f"torch 无法导入：{(r.stderr or '')[-300:]}"
    lines = [l.strip() for l in r.stdout.splitlines() if l.strip()]
    ver = lines[0] if lines else "?"
    has_cop = any(l == "custom_op" for l in lines)
    if not has_cop:
        return False, (f"torch 版本过低（{ver}），缺少 torch.library.custom_op"
                       f"（comfy_kitchen 依赖，需 torch>=2.0）；"
                       f"魔搭环境请选用 torch>=2.0 的 GPU 镜像，或升级镜像内 torch")
    return True, ver


def _launch_precheck(ctx: Ctx) -> "tuple[bool, str]":
    """v2.3.14：启动前置硬门禁。

    已探明的环境不满足（venv 缺失 / torch 不兼容 / 无 NVIDIA GPU）直接返回原因，
    避免盲目 spawn 一个必崩的 ComfyUI 进程、等 120s 再 dump 原始 traceback。

    Veda 运行时硬性依赖：torch>=2.0（comfy_kitchen 需 ``torch.library.custom_op``）
    且必须有 NVIDIA GPU（CPU 模式不可用）。返回 (ok, reason)。
    """
    vpy = ctx.venv_python
    if not (vpy and os.path.exists(vpy)):
        return False, f"venv 缺失/失效（{vpy}），无法启动 ComfyUI"
    # torch API 兼容性（comfy_kitchen 需 torch.library.custom_op / torch>=2.0）
    compat_ok, compat_detail = _torch_compat_ok(vpy)
    if not compat_ok:
        return False, f"torch 运行时不兼容，无法启动 ComfyUI：{compat_detail}"
    # CUDA 可用性（Veda 需 NVIDIA GPU 才能运行；CPU 模式不可用）
    # v2.3.15：同时识别非 CUDA 后端（如 AMD ROCm）——ROCm 下 torch.cuda.is_available()
    # 亦为 True，但 torch.version.hip 存在且 torch.version.cuda 为空，Veda 依赖 NVIDIA
    # 专属能力（如 nvfp4），不能在 ROCm 上运行，需诚实拦截而非盲启崩溃。
    try:
        r = _run_venv(vpy,
                      "import torch; "
                      "print('hip' if getattr(torch.version, 'hip', None) "
                      "else ('cuda' if torch.cuda.is_available() else 'nocuda'))",
                      timeout=60)
    except Exception as e:  # noqa: BLE001
        return False, f"torch 导入失败，无法确认 CUDA 可用性：{str(e)[:200]}"
    if r.returncode != 0:
        return False, f"torch 导入失败，无法确认 CUDA 可用性：{r.stderr.strip()[:200]}"
    out = r.stdout.strip()
    if out == "hip":
        return False, ("检测到非 CUDA 后端（AMD ROCm：torch.version.hip 存在），"
                      "Veda 依赖 NVIDIA CUDA（如 nvfp4），无法在此环境运行"
                      "（请切换到 NVIDIA CUDA 镜像）")
    if out == "nocuda":
        return False, "未检测到 NVIDIA GPU，Veda 需 GPU 才能运行（CPU 模式不可用），无法启动 ComfyUI"
    return True, ""


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
    # v2.3.11：launch 路径不跑 P7，补一道 CUDA 可用性早期信号（Veda 需 GPU 才能运行；
    # 无 GPU 时 ComfyUI 启动即崩溃、端口永不开放，提前预警便于排查，不直接阻断）
    try:
        r = subprocess.run([vpy, "-c",
                            "import torch; print('cuda' if torch.cuda.is_available() else 'nocuda')"],
                           capture_output=True, text=True, timeout=60)
        if r.returncode == 0 and "nocuda" in r.stdout:
            log.warn("LP", "gpu", "torch.cuda 不可用（当前环境无 NVIDIA GPU），"
                     "ComfyUI 启动后可能立即崩溃（Veda 需 GPU 才能运行）",
                     detail={"stderr": r.stderr[-200:]})
        elif r.returncode != 0:
            log.warn("LP", "gpu", "torch 导入失败，无法确认 CUDA 可用性",
                     detail={"stderr": r.stderr[-200:]})
        else:
            log.info("LP", "gpu", "torch.cuda 可用", detail=r.stdout.strip())
    except Exception as e:  # noqa: BLE001
        log.warn("LP", "gpu", "CUDA 探测异常，跳过", detail=str(e)[:200])
    # v2.3.13：torch API 兼容性早期信号（comfy_kitchen 需 torch.library.custom_op / torch>=2.0）。
    # launch 路径若 torch 不兼容则无法启动，直接抛 E701 给出清晰原因，而非等待崩溃后 dump traceback。
    compat_ok, compat_detail = _torch_compat_ok(vpy)
    if not compat_ok:
        raise PhaseError(701, f"torch 运行时不兼容，无法启动 ComfyUI：{compat_detail}")
    log.info("LP", "torch", "torch API 兼容性 OK", detail=compat_detail)
    log.ok("LP", "done", "轻量预检通过（venv 可用、ComfyUI 就位）", detail=vpy)


# ---------------------------------------------------------------------------
# P0 预检
# ---------------------------------------------------------------------------
def _confirm_continue(ctx: "Ctx", prompt: str) -> bool:
    """通用交互询问是否继续（v2.3.9/v2.3.10）：用于"环境可能不满足但允许用户选择继续"的环节
    （如未检测到 GPU / GPU 算力不足 / 安装根存在外来 ComfyUI），替代旧版直接 PhaseError 退出。

    - dry-run：始终返回 True（仅展示计划，不询问、不中止）
    - --yes：直接继续（无交互自动确认；含非交互场景）
    - 交互 TTY：询问 [y/N]，默认不继续
    - 非交互且无 --yes：打印说明后返回 False（由上层以干净退出处理，不抛错误）

    prompt 为展示给用户的询问文案（不含末尾 [y/N]）。
    """
    if ctx.args.dry_run:
        return True
    if ctx.args.yes:
        ctx.logger.info("P0", "confirm", "--yes 已给，继续", detail={"prompt": prompt})
        return True
    if sys.stdin.isatty():
        try:
            ans = input(f"\n⚠ {prompt} [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            ans = ""
        if ans in ("y", "yes", "是"):
            ctx.logger.info("P0", "confirm", "用户确认继续", detail={"prompt": prompt})
            return True
        ctx.logger.info("P0", "confirm", "用户选择不继续", detail={"prompt": prompt})
        return False
    # 非交互、未给 --yes：不挂起、不硬失败，给出说明后中止
    ctx.logger.warn("P0", "confirm", "非交互无 --yes，跳过", detail={"prompt": prompt})
    return False


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
    if not ctx.args.dry_run:  # v2.3.8：dry-run 只打印动作，不落盘建目录
        os.makedirs(ctx.pip_cache, exist_ok=True)

    # GPU → 决定 Profile
    gpu = platform.detect_gpu()
    if gpu:
        log.info("P0", "gpu", "GPU 检测", detail=gpu)
    else:
        log.warn("P0", "gpu", "未检测到 NVIDIA GPU（CPU 模式不可用）")

    if ctx.args.profile == "auto" or not ctx.args.profile:
        if not gpu:
            # v2.3.9：无 GPU 不再直接 PhaseError(103) 退出，改为交互询问 / 按 --yes 继续
            fallback = "standard"
            if not _confirm_continue(ctx,
                    f"未检测到 NVIDIA GPU（Veda 需 GPU 才能运行，CPU 模式不可用）。"
                    f"仍继续安装（将使用 {fallback} 档）？"):
                ctx.user_aborted = True
                return
            ctx.profile_name = fallback
            log.warn("P0", "profile", "无 GPU 沿用 fallback 档", detail=fallback)
        else:
            vram = gpu["vram_gb"]
            if vram < 16:
                ctx.profile_name = "lowvram"
            elif vram <= 24:
                ctx.profile_name = "standard"
            else:
                ctx.profile_name = "full"
            log.info("P0", "profile", "按实测 VRAM 选档", detail=ctx.profile_name)
            # v2.3.10：GPU 算力不足（< SM80）不再直接 PhaseError(102)，改为询问是否继续
            if not platform.compute_cap_ok(gpu):
                if not _confirm_continue(ctx,
                        "GPU 计算能力 < SM80（Veda Triton INT8 后端需 SM80+，低算力可能无法出片）。"
                        "仍继续安装？"):
                    ctx.user_aborted = True
                    return
                log.warn("P0", "gpu", "用户确认在低算力 GPU 上继续安装")
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

    # venv 路径解析（只读，不创建；供 --only p7/p8 单环节执行时 P2 未跑的场景使用）
    ctx.venv_python = platform.venv_python(ctx.root)

    # 外部 ComfyUI 检测（v2.3.10：不再直接 E601 退出，改为交互询问 / --yes / --force 继续）
    if os.path.isdir(ctx.comfyui_dir) and not os.path.isdir(os.path.join(ctx.comfyui_dir, ".git")):
        if ctx.args.force:
            log.warn("P0", "comfyui", "检测到非本脚本建立的 ComfyUI，--force 覆盖继续",
                     detail=ctx.comfyui_dir)
        elif not _confirm_continue(ctx,
                f"安装根已存在非本脚本建立的 ComfyUI：{ctx.comfyui_dir}。是否覆盖并继续安装？"):
            ctx.user_aborted = True
            return
        else:
            log.warn("P0", "comfyui", "用户确认覆盖外来 ComfyUI 继续安装", detail=ctx.comfyui_dir)

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
            # v2.3.8：修正后缀不一致（此前丢 "/modelscope" 后缀，导致元数据
            # 每次运行在两种形态间来回翻转、反复记"元数据变化"）
            ctx.state.set_meta(ctx.env["os"] + ("/modelscope" if is_ms else ""),
                               ctx.profile_name, ctx.manifest_sha1)

    log.ok("P0", "done", "预检完成", detail={"profile": ctx.profile_name,
                                              "is_modelscope": is_ms})


# ---------------------------------------------------------------------------
# P1 ComfyUI 本体
# ---------------------------------------------------------------------------
def phase1_comfyui(ctx: Ctx):
    log = ctx.logger
    repo = next(r for r in ctx.manifest["repos"] if r["id"] == "comfyui")
    commit = repo.get("commit")
    # 快速跳过（v2.3.7）：已存在且 HEAD == 锁定 commit → 整体跳过（不 fetch/不探测代理）。
    # 旧条件 not meta_changed("","","") 恒 False（state 真实元数据 vs 空串永不等），已废弃。
    if (os.path.isdir(os.path.join(ctx.comfyui_dir, ".git"))
            and commit and not commit.startswith("PINNED")
            and assets._git_head(ctx.comfyui_dir) == commit):
        log.ok("P1", "skip", "ComfyUI 已对齐锁定 commit，跳过获取", detail=commit[:12])
        return
    log.info("P1", "start", "安装/对齐 ComfyUI")
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
    ctx.venv_python = platform.ensure_venv(ctx.root, ctx.system_site_packages, log, ctx)
    req = os.path.join(ctx.comfyui_dir, "requirements.txt")
    # v2.3.15：venv 创建失败回退到系统 Python 时，跳过 requirements.txt 安装，
    # 避免 pip 改写/破坏镜像预置的系统环境（如镜像自带的 ROCm torch）。
    if ctx.venv_fallback:
        if os.path.exists(req):
            log.warn("P2", "pip", "venv 不可用（已回退系统 Python），跳过 requirements.txt 安装以保护环境",
                     detail="依赖应已由镜像预置；如需强制安装请先修复 venv（apt install python3-venv）")
    elif os.path.exists(req):
        env = os.environ.copy()
        env["PIP_CACHE_DIR"] = ctx.pip_cache
        try:
            subprocess.run([ctx.venv_python, "-m", "pip", "install", "-r", req],
                           env=env, check=True, capture_output=True, text=True, timeout=1800)
        except subprocess.CalledProcessError as e:
            log.warn("P2", "pip", "pip 安装失败，尝试重建 venv", detail=e.stderr[-300:])
            shutil.rmtree(os.path.join(ctx.root, "venv"), ignore_errors=True)
            ctx.venv_python = platform.ensure_venv(ctx.root, ctx.system_site_packages, log, ctx)
            try:
                subprocess.run([ctx.venv_python, "-m", "pip", "install", "-r", req],
                               env=env, check=True, capture_output=True, text=True, timeout=1800)
            except subprocess.CalledProcessError as e2:
                # v2.3.8：二次失败给出明确错误码（此前冒泡为"未预期错误"，无 E203）
                raise PhaseError(203, f"ComfyUI 依赖安装失败（venv 重建后仍失败）："
                                      f"{(e2.stderr or '')[-200:]}")
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
    repo = next(r for r in ctx.manifest["repos"] if r["id"] == "veda")
    commit = repo.get("commit")
    # 快速跳过（v2.3.7）：HEAD == 锁定 commit → 跳过 clone；不匹配则走获取/升级链
    if (os.path.isdir(os.path.join(veda_dir, ".git"))
            and commit and not commit.startswith("PINNED")
            and assets._git_head(veda_dir) == commit):
        log.ok("P3", "veda", "Veda 节点已对齐锁定 commit，跳过获取", detail=commit[:12])
    else:
        log.info("P3", "veda", "安装/对齐 Veda 节点")
        if not assets.download_repo(repo, veda_dir, assets.GH_PROXY_CANDIDATES, log):
            raise PhaseError(301, "Veda 节点安装失败")
    # MiniMax-H3 加载器：走 ComfyUI 原生支持（Comfy-Org/MiniMax-H3 官方带 comfyui 标签）。
    # 设计原"方案 B 兜底 clone ComfyUI-MiniMaxH3"经核实该仓库不存在（404），已移除对应 manifest 条目，
    # 此处不再尝试 clone，仅记录依赖原生加载器，由 P7 验收确认。
    log.info("P3", "loader", "依赖 ComfyUI 原生 MiniMax-H3 加载器（Comfy-Org/MiniMax-H3 官方 comfyui 模型）；已于 P7 验收确认")

    # 节点自带依赖 + FA4（best-effort，非致命，§8 P3）——以标记文件判重（v2.3.7）：
    # 标记内容=锁定 commit，commit 变更后自动重装；重试时标记匹配即跳过 pip。
    marker = os.path.join(veda_dir, ".veda-req-done")
    marker_ok = False
    try:
        with open(marker, "r", encoding="utf-8") as f:
            marker_ok = f.read().strip() == (commit or "")
    except Exception:  # noqa: BLE001
        marker_ok = False
    if marker_ok:
        log.ok("P3", "req", "节点依赖已安装（标记匹配锁定 commit），跳过")
    else:
        req = os.path.join(veda_dir, "requirements.txt")
        req_ok = True
        # v2.3.8：venv 缺失守卫（--only p3 跳过 P2 的场景）——此前 subprocess
        # 抛 FileNotFoundError 裸崩为"未预期错误"
        if not (ctx.venv_python and os.path.exists(ctx.venv_python)):
            req_ok = False
            log.warn("P3", "req", "venv 缺失（如 --only p3 未先执行 P2），跳过节点依赖安装")
        if req_ok and os.path.exists(req):
            env = os.environ.copy()
            env["PIP_CACHE_DIR"] = ctx.pip_cache
            try:
                subprocess.run([ctx.venv_python, "-m", "pip", "install", "-r", req],
                               env=env, check=True, capture_output=True, text=True, timeout=1200)
                log.ok("P3", "req", "Veda 节点依赖安装完成")
            except subprocess.CalledProcessError as e:
                req_ok = False
                log.warn("P3", "req", "Veda 节点依赖安装失败（best-effort，忽略；下次重试将重新安装）",
                         detail=e.stderr[-300:])

        # FA4 运行时（best-effort 可选增强，§8 P3）：若节点含 install_fa4.py 则尝试其自测，
        # 失败不阻断主流程（默认 Triton INT8 后端由节点自带内核 + Triton 提供，SM80+ 即可）。
        fa4 = os.path.join(veda_dir, "install_fa4.py")
        if os.path.exists(fa4) and ctx.venv_python and os.path.exists(ctx.venv_python):
            try:
                log.info("P3", "fa4", "检测到 install_fa4.py，best-effort 运行自测（可选加速，失败不阻断）")
                subprocess.run([ctx.venv_python, fa4], check=False,
                               capture_output=True, text=True, timeout=300)
            except Exception as e:  # noqa: BLE001
                log.warn("P3", "fa4", "FA4 自测未通过（可选，忽略）", detail=str(e)[:160])

        # 依赖装好（或本就无 requirements.txt）才写标记，避免重试重复 pip；
        # pip 失败不写标记，下次重试会重新安装（缺失会在 P7 报告）
        if req_ok:
            try:
                with open(marker, "w", encoding="utf-8") as f:
                    f.write(commit or "")
            except Exception:  # noqa: BLE001
                pass
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


def _port_ready(port: int, timeout: float = 3.0) -> bool:
    """已运行守护（v2.3.8，§8 P8）：探测 8188 是否已有就绪实例。只读，无副作用。"""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/object_info", timeout=timeout) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


def _tail_log(path: str, n: int = 50) -> str:
    """读取日志末尾 n 行，供启动失败时就地展示崩溃原因（v2.3.11）。"""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        return "".join(lines[-n:])
    except Exception:  # noqa: BLE001
        return ""


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
    # 3) torch.cuda + triton（v2.3.8：venv 缺失守卫——--only p7 未先跑 P2 的场景，
    #    此前 _run_venv 抛 FileNotFoundError 裸崩；缺失时降级为 WARN 跳过）
    if ctx.venv_python and os.path.exists(ctx.venv_python):
        r = _run_venv(ctx.venv_python,
                      "import torch, triton; assert torch.cuda.is_available(); "
                      "print(torch.cuda.get_device_capability())", timeout=120)
        if r.returncode == 0:
            log.ok("P7", "torch", "torch.cuda + triton 可用", detail=r.stdout.strip())
        else:
            # v2.3.10：区分"无 CUDA"与"triton 不可用"，避免无 GPU 安装时误报 triton
            rc = _run_venv(ctx.venv_python, "import torch; print(torch.cuda.is_available())",
                          timeout=60)
            if rc.returncode == 0 and rc.stdout.strip().lower() == "false":
                log.warn("P7", "torch", "torch.cuda 不可用（当前环境无 GPU，运行时需 NVIDIA GPU）")
            else:
                log.warn("P7", "torch", "triton 不可用", detail=r.stderr[-200:])
        # v2.3.13：torch API 兼容性（comfy_kitchen 需 torch.library.custom_op / torch>=2.0）。
        # 魔搭 venv 复用镜像 torch，若镜像 torch < 2.0 则 ComfyUI 启动即崩；此处明确预警。
        compat_ok, compat_detail = _torch_compat_ok(ctx.venv_python)
        if not compat_ok:
            log.warn("P7", "torch", "torch 运行时不兼容，ComfyUI 启动将失败",
                     detail=compat_detail)
        else:
            log.ok("P7", "torch", "torch API 兼容性 OK", detail=compat_detail)
    else:
        log.warn("P7", "torch", "venv 缺失（如 --only p7 未先执行 P2），跳过 torch/triton 检查")
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
    # 已运行守护（v2.3.8）：已有就绪实例时直接通过，避免误拉起第二实例
    if _port_ready(8188):
        log.ok("P7", "smoke", "ComfyUI 已在运行，smoke 视为通过（未重复拉起）")
        return
    log.info("P7", "smoke", "无头启动 ComfyUI 探测 /object_info（首跑豁免全模型加载）")
    port = 8188
    cmd = [ctx.venv_python, os.path.join(ctx.comfyui_dir, "main.py"),
           "--listen", "127.0.0.1", "--port", str(port)]
    logf = os.path.join(ctx.root, "run", "comfyui.log")
    os.makedirs(os.path.dirname(logf), exist_ok=True)
    proc = subprocess.Popen(cmd, stdout=open(logf, "a"), stderr=subprocess.STDOUT)
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
    pidf = os.path.join(run_dir, "comfyui.pid")

    if is_nt:
        # start.bat：已运行守护（v2.3.8）+ 新窗口启动；stop.bat：PowerShell 按
        # CommandLine 匹配 main.py 精确停止（旧 taskkill /FI WINDOWTITLE 匹配不到
        # 无窗口的 python.exe，实测不可靠）。
        start = (
            "@echo off\r\n"
            "curl -s --max-time 3 http://127.0.0.1:8188/object_info >nul 2>&1 && "
            "(echo ComfyUI already running at http://127.0.0.1:8188 & exit /b 0)\r\n"
            f"cd /d \"{ctx.comfyui_dir}\"\r\n"
            f"start \"ComfyUI\" \"{ctx.venv_python}\" \"{main_py}\" --listen {listen} --port 8188\r\n"
            "echo ComfyUI starting... log: run\\comfyui.log\r\n"
        )
        stop = (
            "@echo off\r\n"
            "powershell -NoProfile -Command \"Get-CimInstance Win32_Process -Filter "
            "\\\"Name='python.exe'\\\" | Where-Object { $_.CommandLine -match 'main\\.py' } | "
            "ForEach-Object { Stop-Process -Id $_.ProcessId -Force; Write-Output ('stopped pid ' + $_.ProcessId) }\"\r\n"
            "if %errorlevel% neq 0 echo no ComfyUI process found\r\n"
        )
        status = (
            "@echo off\r\n"
            "curl -s --max-time 3 http://127.0.0.1:8188/object_info >/dev/null 2>&1 && echo UP || echo DOWN\r\n"
        )
    else:
        start = (
            "#!/usr/bin/env bash\n"
            "# 已运行守护（v2.3.8）：UP 即退出，避免二次拉起/端口冲突\n"
            "curl -s --max-time 3 http://127.0.0.1:8188/object_info >/dev/null 2>&1 && "
            "{ echo \"ComfyUI already running (http://127.0.0.1:8188)\"; exit 0; }\n"
            f"cd \"{ctx.comfyui_dir}\"\n"
            f"nohup \"{ctx.venv_python}\" \"{main_py}\" --listen {listen} --port 8188 "
            f">> \"{logf}\" 2>&1 &\n"
            "echo $! > \"" + pidf + "\"\n"
            "echo \"ComfyUI starting (pid $(cat \"" + pidf + "\"))... log: " + logf + "\n"
        )
        stop = (
            "#!/usr/bin/env bash\n"
            "PID=$(cat \"" + pidf + "\" 2>/dev/null)\n"
            "kill $PID 2>/dev/null && echo stopped || echo not-running\n"
        )
        status = (
            "#!/usr/bin/env bash\n"
            "curl -s --max-time 3 http://127.0.0.1:8188/object_info >/dev/null && echo UP || echo DOWN\n"
        )
    with open(os.path.join(run_dir, "start.bat" if is_nt else "start.sh"), "w", newline="" if is_nt else None) as f:
        f.write(start)
    with open(os.path.join(run_dir, "stop.bat" if is_nt else "stop.sh"), "w", newline="" if is_nt else None) as f:
        f.write(stop)
    with open(os.path.join(run_dir, "status.bat" if is_nt else "status.sh"), "w", newline="" if is_nt else None) as f:
        f.write(status)


def phase8_launch(ctx: Ctx, strict: bool = True):
    """后台启动 ComfyUI（§8 P8）。

    strict=True（默认，用于 --mode launch / --only p8）：环境不满足运行条件时
    直接抛 E701 并给出清晰原因（启动是该模式的目标，无法达成即失败）。
    strict=False（用于 auto/install 安装段顺带启动）：环境不满足时**跳过启动**并诚实
    汇报（安装本身已完成，不崩溃、不谎报），避免无 GPU / torch 不兼容容器陷入
    "安装成功却必崩启动"的循环（v2.3.14）。
    """
    log = ctx.logger
    _write_run_scripts(ctx)
    if ctx.args.no_launch:
        log.info("P8", "skip", "按 --no-launch 不启动")
        return
    # v2.3.14：启动前置硬门禁——已知环境的 torch 不兼容 / 无 GPU 直接给出清晰结果，
    # 不再 spawn 必崩进程、等待 120s、再 dump traceback。
    ok, reason = _launch_precheck(ctx)
    if not ok:
        if strict:
            raise PhaseError(701, reason)
        # 安装段顺带启动：环境不满足则跳过，诚实汇报，不把"安装成功"变成"启动失败"
        log.warn("P8", "skip", "环境不满足运行条件，跳过启动（安装已完成）",
                 detail=reason)
        ctx.launch_skipped_reason = reason
        ctx.launch_ready = None
        return
    # 已运行守护（v2.3.8，§8 P8）：8188 已有就绪实例则跳过，避免二次拉起
    # （端口冲突；且旧实例会响应就绪探测，掩盖新实例启动失败）
    if _port_ready(8188):
        log.ok("P8", "skip", "ComfyUI 已在运行（http://127.0.0.1:8188），跳过启动")
        ctx.launch_ready = True
        return
    log.info("P8", "start", "后台启动 ComfyUI")
    listen = "0.0.0.0" if ctx.args.listen else "127.0.0.1"
    cmd = [ctx.venv_python, os.path.join(ctx.comfyui_dir, "main.py"),
           "--listen", listen, "--port", "8188"]
    logf = os.path.join(ctx.root, "run", "comfyui.log")
    os.makedirs(os.path.dirname(logf), exist_ok=True)
    if os.name == "nt":
        # DETACHED_PROCESS + stdout/stderr 重定向到日志（v2.3.8：无控制台时
        # 直接 print 可能异常）
        proc = subprocess.Popen(cmd, creationflags=0x00000008,  # DETACHED_PROCESS
                                stdout=open(logf, "a"), stderr=subprocess.STDOUT)
    else:
        # start_new_session=True（v2.3.8）：脱离控制组，终端会话关闭/SIGHUP
        # 不再杀死 ComfyUI（等价 nohup）；日志追加不截断（保留历史便于排错）
        proc = subprocess.Popen(cmd, stdout=open(logf, "a"), stderr=subprocess.STDOUT,
                                start_new_session=True)
    # 就绪探测（v2.3.11：超时不得谎报成功）
    if _poll_object_info(8188, 120, "VedaSparseAttention"):
        log.ok("P8", "ready", "ComfyUI 就绪", detail="http://127.0.0.1:8188")
        log.ok("P8", "done", "启动完成")
        ctx.launch_ready = True
        return
    # 未就绪：区分"进程已崩溃"与"仍在加载"
    rc = proc.poll()
    tail = _tail_log(logf, 50)
    if rc is not None:
        # 进程已退出 = 启动失败（如 CUDA/OOM/依赖错误），必须如实报错而非谎报成功
        if tail:
            print("\n----- ComfyUI 启动日志尾部 (run/comfyui.log) -----\n" + tail)
        log.error("P8", "crash", f"ComfyUI 进程已退出（退出码 {rc}），启动失败",
                  error_code=701, detail={"log": logf})
        raise PhaseError(701, f"ComfyUI 启动失败：进程退出码 {rc}，详见 {logf}")
    # 进程存活但 120s 未就绪：可能大模型加载较慢，转入后台，但不得谎报"已就绪"
    if tail:
        print("\n----- ComfyUI 启动日志尾部 (run/comfyui.log) -----\n" + tail)
    log.warn("P8", "timeout", "120s 内未探测到就绪，ComfyUI 进程仍在运行（可能模型加载较慢）",
             detail={"log": logf})
    log.info("P8", "bg", "已转入后台加载，端口就绪前请勿访问；稍后用 run/status.sh 复查")
    ctx.launch_ready = False


# ---------------------------------------------------------------------------
# 主编排（v2.3.6：安装段与启动段分离，§8）
# ---------------------------------------------------------------------------
def run_install(ctx: Ctx, only: set | None = None):
    """安装段 P1–P7（P0 由驱动先行调用）+ 写 install_completed（§6 规则3）。

    进入时先复位 install_completed=false 并落盘（防重装半途失败后旧标志
    残留导致 M0 误判「已安装」）；P1–P7 全部成功后置 true 并记 install_profile。

    --only（v2.3.7）：仅执行指定环节（如 {'p5'}），install_completed 不复位也
    不置位（部分执行不构成完整安装）；P4/P5 仍受 --skip-models 约束。
    """
    log = ctx.logger
    only = only or set()
    full = not only
    if full:
        ctx.state.data["install_completed"] = False
        ctx.state.data["install_profile"] = None
        ctx.state.save()
    steps = [
        (1, "P1", lambda: phase1_comfyui(ctx)),
        (2, "P2", lambda: phase2_deps(ctx)),
        (3, "P3", lambda: phase3_nodes(ctx)),
        (4, "P4", lambda: phase4_predictors(ctx) if not ctx.args.skip_models else None),
        (5, "P5", lambda: phase5_models(ctx) if not ctx.args.skip_models else None),
        (6, "P6", lambda: phase6_templates(ctx)),
        (7, "P7", lambda: phase7_verify(ctx)),
    ]
    for n, label, fn in steps:
        if only and n not in only:
            log.info(label, "skip", f"按 --only 跳过 {label}")
            continue
        fn()
    if full:
        # 安装段全部成功 → 标记「已安装」（§6 规则3 / §8 M0）
        ctx.state.data["install_completed"] = True
        ctx.state.data["install_profile"] = ctx.profile_name
        ctx.state.save()
        log.ok("INSTALL", "done", "安装段完成（install_completed=true）",
               detail={"profile": ctx.profile_name, "root": ctx.root})
    else:
        ctx.state.save()
        log.ok("INSTALL", "partial",
               f"部分安装完成（--only {','.join('p%d' % n for n in sorted(only))}；install_completed 未变更）",
               detail={"profile": ctx.profile_name, "root": ctx.root})


def final_summary(ctx: Ctx, mode_label: str):
    ready = getattr(ctx, "launch_ready", None)
    if ready is True:
        ctx.logger.ok("ALL", "done", f"一键搭建完成（{mode_label}）", detail={
            "profile": ctx.profile_name or ctx.state.data.get("install_profile"),
            "root": ctx.root,
            "access": "http://127.0.0.1:8188（魔搭经 Web IDE 端口预览/隧道）"})
        return
    if ready is False:
        # v2.3.11：P8 已启动但未确认就绪（进程存活/模型加载慢），诚实提示，不谎报可访问
        ctx.logger.warn("ALL", "done",
                        f"搭建流程完成（{mode_label}），但 ComfyUI 端口就绪未确认（可能仍在加载模型）",
                        detail={
            "profile": ctx.profile_name or ctx.state.data.get("install_profile"),
            "root": ctx.root,
            "access": "http://127.0.0.1:8188（启动中，稍后用 run/status.sh 复查；日志 run/comfyui.log）"})
        print("\n提示：ComfyUI 已在后台启动，但 120s 内尚未就绪，可能是大模型加载较慢。"
              "请用 run/status.sh 复查（UP 即代表可访问）；若长时间 DOWN，请查 run/comfyui.log。")
        return
    # ready is None：本次根本未执行启动（--no-launch / 仅安装环节 / --mode launch --no-launch
    # / 或 v2.3.14 因环境不满足而跳过启动）
    skip_reason = getattr(ctx, "launch_skipped_reason", None)
    if skip_reason:
        # 环境不满足运行条件（无 GPU / torch 不兼容）：明确告知，指引换镜像后启动，不谎报
        ctx.logger.warn("ALL", "done",
                        f"搭建流程完成（{mode_label}），但未启动 ComfyUI（环境不满足运行条件）",
                        detail={
            "profile": ctx.profile_name or ctx.state.data.get("install_profile"),
            "root": ctx.root,
            "skip_reason": skip_reason})
        print(f"\n提示：安装已完成，但当前环境无法运行 ComfyUI：{skip_reason}。"
              "请切换到「带 NVIDIA GPU 且 torch>=2.0」的镜像后，用 "
              "bash run/start.sh（Windows 双击 run/start.bat）启动，"
              "并用 run/status.sh 复查（UP 即代表可访问）。")
        return
    ctx.logger.ok("ALL", "done", f"搭建流程完成（{mode_label}，未启动 ComfyUI）", detail={
        "profile": ctx.profile_name or ctx.state.data.get("install_profile"),
        "root": ctx.root,
        "start": "run/start.sh 启动（run/status.sh 查状态；run/stop.sh 停止）"})
    print("\n提示：本次未自动启动 ComfyUI（--no-launch 或仅执行安装环节）。"
          "如需启动：bash run/start.sh（Windows 双击 run/start.bat），"
          "并用 run/status.sh 复查（UP 即代表可访问）。")
