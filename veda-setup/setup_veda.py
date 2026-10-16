#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Veda-on-ComfyUI 一键搭建 · 主驱动（跨平台，仅标准库）。

人工唯一入口：python setup_veda.py （或经 install.sh / install.bat）。
设计文档：docs/Veda一键脚本设计文档.md（v2.3.15）。

运行模式（M0，默认 auto）：自动判定是否已安装——
已安装实例仅启动（跳过 P1–P7 安装段）；未安装则安装+启动。
可用 --mode install / --mode launch 显式控制；--only pN[,pN] 单环节执行。
"""
from __future__ import annotations

import os
import sys
import json
import argparse

# 控制台 UTF-8（Windows CP936 防乱码，§7.5）
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore
except Exception:  # noqa: BLE001
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import diag, platform, phases, assets  # noqa: E402

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="setup_veda.py",
        description="Veda-on-ComfyUI 一键搭建（魔搭 Notebook / Windows / Linux）")
    p.add_argument("--install-root", help="安装根目录（默认自动探测）")
    p.add_argument("--mode", choices=["auto", "install", "launch"], default="auto",
                   help="运行模式：auto=自动判定(默认，已安装仅启动/未安装安装+启动)；"
                        "install=强制进入安装段（幂等增量）；launch=强制仅启动（未完成报 E701）")
    p.add_argument("--only", metavar="pN[,pN]",
                   help="仅执行指定安装环节（v2.3.7）：p1–p7 逗号分隔，如 --only p5（补模型）"
                        "或 --only p1,p3（重装 ComfyUI+节点）；P0 预检始终前置；"
                        "install_completed 不变更；未含 p8 则不启动")
    p.add_argument("--profile", default="auto",
                   help="minimal/standard/full/lowvram/auto")
    p.add_argument("--source-priority", default="auto",
                   help="下载源优先级（默认 auto 按环境自适应）")
    p.add_argument("--skip-models", action="store_true", help="跳过主模型资产")
    p.add_argument("--no-accept-license", dest="accept_license", action="store_false",
                   default=True,
                   help="关闭自动接受 License（默认自动接受；关闭后非交互模式将拒绝 E403）")
    p.add_argument("--accept-third-party", action="store_true",
                   help="允许下载社区第三方量化权重（lowvram 档）")
    p.add_argument("--local-code", action="store_true",
                   help="代码+venv 放本地盘（魔搭环境强制 OFF，仅 T2/T3 可选）")
    p.add_argument("--keep-cache", action="store_true",
                   help="下载后保留模型缓存（默认清理省磁盘）")
    p.add_argument("--listen", action="store_true", help="绑定 0.0.0.0（默认 127.0.0.1）")
    p.add_argument("--interactive", action="store_true", help="退回人工确认模式")
    p.add_argument("-y", "--yes", action="store_true",
                   help="自动确认所有交互提示（含无 GPU 时继续安装，不再询问直接继续）")
    p.add_argument("--dry-run", action="store_true", help="只打印动作不落盘")
    p.add_argument("--resume", action="store_true", help="显式续跑（等价幂等重跑）")
    p.add_argument("--deep-verify", action="store_true", help="资产全量哈希校验")
    p.add_argument("--smoke-test", action="store_true", help="无头启动探测 /object_info")
    p.add_argument("--no-launch", action="store_true", help="装完不启动 ComfyUI")
    p.add_argument("--doctor", action="store_true", help="只生成诊断包后退出")
    p.add_argument("--log-dir", help="日志目录（默认 <root>/logs）")
    p.add_argument("--force", action="store_true", help="覆盖外来 ComfyUI 检测")
    return p.parse_args()


def parse_only_spec(spec: str) -> set:
    """解析 --only 参数（v2.3.7）：'p5' / '5' / 'p1,p3' → {5} / {1,3}。

    合法环节 1–7（p8 由启动逻辑单独处理：含 8 时执行后启动）；非法值抛 ValueError。
    """
    out = set()
    for tok in spec.replace(" ", "").split(","):
        if not tok:
            continue
        t = tok.lower().lstrip("p")
        if not t.isdigit():
            raise ValueError(f"无法识别的环节：{tok}（应为 p1–p8）")
        n = int(t)
        if not 1 <= n <= 8:
            raise ValueError(f"环节超出范围：{tok}（应为 p1–p8）")
        out.add(n)
    if not out:
        raise ValueError("--only 为空")
    return out


def resolve_root(args: argparse.Namespace) -> str:
    if args.install_root:
        return os.path.abspath(args.install_root)
    env = os.environ.get("COMFYUI_INSTALL_ROOT")
    if env:
        return os.path.abspath(env)
    if os.path.isdir("/mnt/workspace"):
        return "/mnt/workspace/comfyui-veda"
    if os.name == "nt":
        return os.path.join(SCRIPT_DIR, "ComfyUI")
    return os.path.join(os.path.expanduser("~"), "comfyui-veda")


def version_snapshot(ctx) -> dict:
    snap = {}
    try:
        snap["modelscope"] = os.popen("modelscope --version 2>&1").read().strip()[:60]
    except Exception:  # noqa: BLE001
        snap["modelscope"] = "n/a"
    try:
        snap["pip"] = os.popen(f"{sys.executable} -m pip --version 2>&1").read().strip()[:60]
    except Exception:  # noqa: BLE001
        snap["pip"] = "n/a"
    # MiniMax-H3 加载器走 ComfyUI 原生（ComfyUI-MiniMaxH3 仓库不存在，已废弃该引用）
    dirs = {
        "ComfyUI": ctx.comfyui_dir,
        "Veda": os.path.join(ctx.comfyui_dir, "custom_nodes", "Veda-on-ComfyUI"),
    }
    for name, d in dirs.items():
        if d and os.path.isdir(os.path.join(d, ".git")):
            try:
                out = os.popen(f'git -C "{d}" rev-parse HEAD 2>&1').read().strip()
                snap[name] = out[:12]
            except Exception:  # noqa: BLE001
                snap[name] = "n/a"
        else:
            snap[name] = "not-installed"
    return snap


def do_doctor(args: argparse.Namespace, root: str, log_dir: str, logger: diag.Logger) -> int:
    logger.info("DOCTOR", "start", "生成诊断包")
    net = platform.network_probe(platform.DEFAULT_HOSTS, logger)
    gpu = platform.detect_gpu()
    snap = version_snapshot_dummy(root)
    config = {"profile": args.profile, "install_root": root,
              "is_modelscope": platform.is_modelscope(), "gpu": gpu}
    arc = diag.diagnose(root if os.path.isdir(root) else os.getcwd(), log_dir,
                        {}, config, net, snap)
    logger.ok("DOCTOR", "done", "诊断包已生成", detail=arc)
    print(f"\n诊断包：{arc}\n可附此文件提 issue。")
    return 0


def version_snapshot_dummy(root: str) -> dict:
    return {"note": "doctor 模式（无完整 ctx）"}


def main() -> int:
    args = parse_args()

    if os.name == "nt" and not sys.stdout.encoding:
        pass

    root = resolve_root(args)
    log_dir = args.log_dir or os.path.join(root, "logs")
    os.makedirs(log_dir, exist_ok=True)
    logger = diag.Logger(log_dir)

    if args.doctor:
        return do_doctor(args, root, log_dir, logger)

    # 加载 manifest / profiles
    manifest_path = os.path.join(SCRIPT_DIR, "assets_manifest.json")
    profiles_path = os.path.join(SCRIPT_DIR, "profiles.json")
    try:
        manifest = load_json(manifest_path)
        profiles = load_json(profiles_path)
    except Exception as e:  # noqa: BLE001
        logger.error("INIT", "load", "加载 manifest/profiles 失败", detail=str(e)[:200])
        return 1
    manifest_sha1 = assets.compute_manifest_sha1(manifest_path)

    # 平台识别 + 魔搭强制 --local-code OFF（E3）
    env = platform.detect_platform()
    if env["is_modelscope"] and args.local_code:
        logger.warn("P0", "local-code", "魔搭环境强制 --local-code OFF（§6 规则7）")
        args.local_code = False
    if args.source_priority != "auto":
        logger.info("P0", "source", "自定义源链覆盖（当前仍按 auto 自适应为主）",
                    detail=args.source_priority)

    # 并发锁（含 stale 释放）
    lock = platform.FileLock(os.path.join(root, ".setup.lock"))
    if not lock.acquire(logger):
        arc = diag.diagnose(root if os.path.isdir(root) else os.getcwd(), log_dir, {},
                            {"stage": "lock"}, {}, {})
        logger.error("P0", "lock", "并发锁冲突（另一安装进行中）", error_code=603)
        print(f"错误码 E603：另一安装正在进行，或锁文件残留。诊断包：{arc}")
        return 603

    ctx = phases.Ctx()
    ctx.root = root
    ctx.comfyui_dir = os.path.join(root, "ComfyUI")
    ctx.manifest = manifest
    ctx.manifest_sha1 = manifest_sha1
    ctx.manifest_profiles = profiles["profiles"]
    ctx.state = diag.State(root)
    ctx.logger = logger
    ctx.args = args
    ctx.config = {"profile": args.profile, "install_root": root,
                  "is_modelscope": env["is_modelscope"], "source_priority": args.source_priority}

    try:
        # --only 单环节执行（v2.3.7）：跳过 M0 分流，强制安装段路径但仅执行指定环节
        only_set = parse_only_spec(args.only) if args.only else None
    except ValueError as e:
        logger.error("INIT", "args", str(e))
        print(f"参数错误：{e}")
        return 1
    if only_set and args.mode == "launch":
        logger.warn("M0", "mode", "--only 与 --mode launch 同用，以 --only 为准（走安装段指定环节）")

    try:
        # M0 运行模式判定（§8）：先读 state，再决定 install / launch
        ctx.state.load()

        if only_set:
            logger.info("M0", "mode", "MODE_DECISION",
                        detail={"decision": "install-partial",
                                "reason": f"--only {','.join('p%d' % n for n in sorted(only_set))}",
                                "forced": args.mode, "is_installed": None})
            if args.dry_run:
                phases.phase0_preflight(ctx)
                print(f"\n[DRY-RUN] 安装根={root}  模式=install-partial（--only "
                      f"{','.join('p%d' % n for n in sorted(only_set))}）")
                print(f"[DRY-RUN] 选定 Profile={ctx.profile_name}  魔搭={env['is_modelscope']}"
                      f"  源链={ctx.network_table}")
                print(f"[DRY-RUN] 将执行：P0 预检 → "
                      f"{' → '.join('P%d' % n for n in sorted(only_set))}"
                      + (" → P8 启动" if 8 in only_set else "（不含 P8，不启动）"))
                return 0
            phases.phase0_preflight(ctx)
            if ctx.user_aborted:  # v2.3.9/v2.3.10：无 GPU / 算力不足 / 外来 ComfyUI 经询问后用户选择不继续
                logger.rotate()
                print("\n已取消安装：环境预检未通过且用户选择不继续"
                      "（未检测到 GPU / GPU 算力不足 / 安装根存在外来 ComfyUI 等）。"
                      "（如需继续请加 --yes，或用 --profile 显式指定档位 / --force 覆盖外来 ComfyUI）")
                return 0
            phases.run_install(ctx, only=only_set)
            if 8 in only_set and not args.no_launch:
                # v2.3.8：venv 守卫（--only p8 单独执行且 P2 未跑/安装不完整的场景），
                # 此前 Popen 抛 FileNotFoundError 裸崩为"未预期错误"
                if not (ctx.venv_python and os.path.exists(ctx.venv_python)):
                    logger.error("P8", "launch", "--only p8 但 venv 缺失（先执行 --only p2 或完整安装）",
                                 error_code=701)
                    print("\n失败：错误码 E701 (701)")
                    print("原因：--only p8 但 venv 缺失——请先 python setup_veda.py --only p2"
                          "（建 venv/依赖）或直接完整安装（默认 auto）")
                    return 701
                phases.phase8_launch(ctx, strict=True)
            else:
                logger.info("P8", "skip", "--only 未包含 p8（或 --no-launch），不启动")
            logger.rotate()
            phases.final_summary(ctx, f"install-partial：--only {','.join('p%d' % n for n in sorted(only_set))}")
            return 0

        mode = phases.decide_mode(ctx)

        if args.dry_run:
            if mode == "launch":
                print(f"\n[DRY-RUN] 安装根={root}  模式=launch-only（已安装，将跳过 P1–P7）")
                print("[DRY-RUN] 将执行：轻量预检 → P8 启动")
            else:
                phases.phase0_preflight(ctx)
                print(f"\n[DRY-RUN] 安装根={root}  模式=install（安装+启动）  "
                      f"选定 Profile={ctx.profile_name}  "
                      f"魔搭={env['is_modelscope']}  源链={ctx.network_table}")
                print("[DRY-RUN] 将执行：P1 ComfyUI → P2 deps → P3 节点 → "
                      "P4 预测器 → P5 主模型 → P6 模板 → P7 验收 → P8 启动")
            return 0

        if mode == "launch":
            try:
                phases.preflight_launch(ctx)
            except phases.PhaseError as e:
                if e.code == 701 and args.mode == "auto":
                    # auto 降级：转安装段（幂等增量，如仅重建失效 venv），不报错（§8 轻量预检）
                    logger.warn("M0", "fallback",
                                "launch 轻量预检失败，auto 自动降级转安装段",
                                detail={"fallback_reason": "launch-preflight-failed",
                                        "error": e.msg[:200]})
                    mode = "install"
                else:
                    raise
            else:
                if args.no_launch:
                    logger.warn("P8", "skip",
                                "--mode launch 与 --no-launch 同用，跳过启动（仅确认已安装）")
                    logger.rotate()
                    phases.final_summary(ctx, "已安装，未启动（--no-launch）")
                    return 0
                phases.phase8_launch(ctx, strict=True)
                logger.rotate()
                phases.final_summary(ctx, "launch-only：已安装实例仅启动")
                return 0

        # mode == "install"（auto 未安装 / 显式 install / auto 由 launch 降级）
        phases.phase0_preflight(ctx)
        if ctx.user_aborted:  # v2.3.9/v2.3.10：无 GPU / 算力不足 / 外来 ComfyUI 经询问后用户选择不继续
            logger.rotate()
            print("\n已取消安装：环境预检未通过且用户选择不继续"
                  "（未检测到 GPU / GPU 算力不足 / 安装根存在外来 ComfyUI 等）。"
                  "（如需继续请加 --yes，或用 --profile 显式指定档位 / --force 覆盖外来 ComfyUI）")
            return 0
        phases.run_install(ctx)
        if args.no_launch:
            logger.info("P8", "skip", "按 --no-launch 不启动")
        else:
            phases.phase8_launch(ctx, strict=False)
        logger.rotate()
        phases.final_summary(ctx, "install：安装+启动")
        return 0
    except phases.PhaseError as e:
        logger.error("FATAL", "phase", e.msg, error_code=e.code)
        arc = diag.diagnose(root, log_dir, ctx.state.data, ctx.config,
                            ctx.network_table, version_snapshot(ctx))
        print(f"\n失败：错误码 {diag.error_name(e.code)} ({e.code})")
        print(f"原因：{e.msg}")
        print(f"日志：{logger.jsonl_path}")
        print(f"诊断包：{arc}")
        return e.code
    except Exception as e:  # noqa: BLE001
        logger.error("FATAL", "unexpected", str(e)[:300])
        arc = diag.diagnose(root, log_dir, ctx.state.data, ctx.config,
                            ctx.network_table, version_snapshot(ctx))
        print(f"\n未预期错误：{e}")
        print(f"日志：{logger.jsonl_path}")
        print(f"诊断包：{arc}")
        return 1
    finally:
        lock.release()


if __name__ == "__main__":
    sys.exit(main())
