"""assets.py — 资产下载引擎：模型权重（modelscope/hf）+ repo（git→tarball 降级）。

源链自适应（§9.2）：魔搭实例排除 hf-mirror/huggingface，通用环境保留。
所有下载直落目标目录（不落 /tmp 再拷贝）。校验三件套：存在/大小/头部 JSON。
"""
from __future__ import annotations

import os
import sys
import json
import struct
import hashlib
import shutil
import subprocess
import tempfile

# GitHub 代理候选（多候选 + 探测，禁止带 Token，§9.2 / E6）
# 候选来源：2026-10-12 实测可达 gh-proxy.com / ghproxy.net / gitclone.com（均 200）。
# mirror.ghproxy.com 实测不可达（HTTP 000）已废弃，不纳入候选链。
GH_PROXY_CANDIDATES = [
    "https://gh-proxy.com/https://github.com/",
    "https://ghproxy.net/https://github.com/",
    "https://gitclone.com/https://github.com/",
]


def compute_manifest_sha1(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# 校验 / 指纹
# ---------------------------------------------------------------------------
def verify_safetensors_header(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            hlen = struct.unpack("<Q", f.read(8))[0]
            if not (0 < hlen < 100_000_000):
                return False
            json.loads(f.read(hlen))
        return True
    except Exception:  # noqa: BLE001
        return False


def verify_file(path: str, min_bytes: int = 0, check_header: bool = False) -> bool:
    if not os.path.exists(path):
        return False
    try:
        sz = os.path.getsize(path)
    except OSError:
        return False
    if sz < min_bytes:
        return False
    if check_header and path.endswith(".safetensors"):
        return verify_safetensors_header(path)
    return True


def fingerprint(path: str) -> str:
    sz = os.path.getsize(path)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(65536))
    return f"{sz}:{h.hexdigest()[:16]}"


# ---------------------------------------------------------------------------
# 源链解析（模型权重）
# ---------------------------------------------------------------------------
def resolve_model_chain(asset: dict, network_table: dict, is_modelscope: bool,
                        gh_proxy_ok: bool) -> list:
    """返回按优先级排序的源 dict 列表（已针对环境过滤）。"""
    sources = asset.get("sources", [])
    chain = []
    if is_modelscope:
        chain += [s for s in sources if s.get("type") == "modelscope"]
        # best-effort：gh-proxy 可达时，尝试经 gh-proxy 代理 huggingface（G2/G5 未定，仅尝试）
        if gh_proxy_ok:
            for s in sources:
                if s.get("type") == "hf":
                    s2 = dict(s)
                    s2["endpoint"] = "https://gh-proxy.com/https://huggingface.co"
                    chain.append(s2)
    else:
        chain += [s for s in sources if s.get("type") in ("modelscope", "hf")]
    return chain


# ---------------------------------------------------------------------------
# 工具：确保 CLI 可用
# ---------------------------------------------------------------------------
def _ensure_modelscope(env: dict, logger=None):
    try:
        subprocess.run(["modelscope", "--version"], env=env, capture_output=True, timeout=30)
    except Exception:  # noqa: BLE001
        if logger:
            logger.info("P5", "dep", "安装 modelscope CLI")
        subprocess.run([sys.executable, "-m", "pip", "install", "modelscope"],
                       env=env, check=True, capture_output=True, text=True, timeout=600)


def _ensure_hf_cli(env: dict, logger=None):
    try:
        subprocess.run(["huggingface-cli", "--version"], env=env, capture_output=True, timeout=30)
    except Exception:  # noqa: BLE001
        if logger:
            logger.info("P5", "dep", "安装 huggingface_hub[cli]")
        subprocess.run([sys.executable, "-m", "pip", "install", "-U", "huggingface_hub[cli]"],
                       env=env, check=True, capture_output=True, text=True, timeout=600)


# ---------------------------------------------------------------------------
# 模型权重下载
# ---------------------------------------------------------------------------
def _dl_modelscope(repo: str, include: str, local_dir: str, cache_dir: str | None,
                   logger=None, timeout: int = 1800):
    os.makedirs(local_dir, exist_ok=True)
    env = os.environ.copy()
    if cache_dir:
        env["MODELSCOPE_CACHE"] = cache_dir
    _ensure_modelscope(env, logger)
    cmd = ["modelscope", "download", "--model", repo,
           "--local_dir", local_dir, "--include", include]
    subprocess.run(cmd, env=env, check=True, capture_output=True, text=True, timeout=timeout)


def _dl_hf(repo: str, include: str, local_dir: str, endpoint: str | None,
           cache_dir: str | None, logger=None, timeout: int = 1800):
    os.makedirs(local_dir, exist_ok=True)
    env = os.environ.copy()
    if endpoint:
        env["HF_ENDPOINT"] = endpoint
    if cache_dir:
        env["HF_HOME"] = cache_dir
    _ensure_hf_cli(env, logger)
    cmd = ["huggingface-cli", "download", repo,
           "--include", include, "--local-dir", local_dir]
    subprocess.run(cmd, env=env, check=True, capture_output=True, text=True, timeout=timeout)


def download_model_asset(asset: dict, root: str, chain: list, env: dict,
                         keep_cache: bool, logger, state) -> tuple[bool, str | None]:
    target_rel = asset["target"]
    target = os.path.join(root, *target_rel.split("/"))
    fp = fingerprint(target) if os.path.exists(target) else None
    if fp and state.is_asset_done(asset["id"], fp):
        logger.ok("P5", "asset", f"跳过已完成 {asset['id']}")
        return True, None
    local_dir = os.path.join(root, "ComfyUI", "models")
    include = asset.get("include") or asset.get("file") or os.path.basename(target_rel)
    last_err = None
    for src in chain:
        try:
            if src.get("type") == "modelscope":
                _dl_modelscope(src["repo"], include, local_dir, env.get("modelscope_cache"), logger)
            elif src.get("type") == "hf":
                _dl_hf(src["repo"], include, local_dir, src.get("endpoint"),
                       env.get("hf_cache"), logger)
            else:
                continue
            if verify_file(target, asset.get("min_bytes", 0), check_header=True):
                state.record_asset(asset["id"], "ok", os.path.getsize(target),
                                   fingerprint(target), src.get("type"))
                logger.ok("P5", "asset", f"下载成功 {asset['id']}", detail=src.get("type"))
                return True, src.get("type")
            logger.warn("P5", "asset", f"校验失败 {asset['id']}", detail=src.get("type"))
        except Exception as e:  # noqa: BLE001
            last_err = f"{src.get('type')}: {e}"
            logger.warn("P5", "asset", f"源失败 {asset['id']}", detail=str(e)[:160])
    if last_err:
        logger.error("P5", "asset", f"所有源失败 {asset['id']}", detail=last_err)
    return False, None


def cleanup_cache(cache_dir: str, logger=None):
    if cache_dir and os.path.isdir(cache_dir):
        try:
            shutil.rmtree(cache_dir, ignore_errors=True)
            if logger:
                logger.info("P5", "cache", "清理模型缓存", detail=cache_dir)
        except Exception as e:  # noqa: BLE001
            if logger:
                logger.warn("P5", "cache", "清理缓存失败", detail=str(e)[:120])


# ---------------------------------------------------------------------------
# repo 下载（git → tarball 降级，§8 P1/P3 / E6）
# ---------------------------------------------------------------------------
def _looks_like_sha(commit: str) -> bool:
    return len(commit) == 40 and all(c in "0123456789abcdef" for c in commit.lower())


def _git_clone(url: str, commit: str | None, dest: str, prox: str | None,
               logger=None, timeout: int = 600):
    """克隆/检出仓库到 dest，并精准对齐到 commit（分支名或 40 位 SHA）。

    关键修正（§8 P1）：SHA 钉死不能用 `git clone --depth 1 --branch <sha>`
    再 `checkout <sha>` —— 浅克隆不含历史提交，checkout 会失败。正确做法：
    blobless clone（不检出）→ fetch origin <sha> → checkout <sha>。
    分支/标签走浅克隆（--depth 1 --branch）更轻量。
    prox 通过 `-c url.<prox>.insteadOf` 临时重写，不污染全局/仓库配置。
    """
    prefix = ["git"]
    if prox:
        prefix += ["-c", f'url."{prox}".insteadOf="https://github.com/"']

    if os.path.isdir(os.path.join(dest, ".git")):
        # 已存在：直接 fetch + checkout（兼容分支名或 SHA）
        if commit and not commit.startswith("PINNED"):
            subprocess.run(prefix + ["-C", dest, "fetch", "origin", commit],
                           check=True, capture_output=True, text=True, timeout=timeout)
            subprocess.run(["git", "-C", dest, "checkout", commit],
                           check=True, capture_output=True, text=True, timeout=120)
        return

    if commit and not commit.startswith("PINNED"):
        if _looks_like_sha(commit):
            # SHA 钉死：blobless clone（不检出）→ fetch 指定 SHA → checkout
            subprocess.run(prefix + ["clone", "--filter=blob:none", "--no-checkout",
                                     url, dest],
                           check=True, capture_output=True, text=True, timeout=timeout)
            subprocess.run(prefix + ["-C", dest, "fetch", "origin", commit],
                           check=True, capture_output=True, text=True, timeout=timeout)
            subprocess.run(["git", "-C", dest, "checkout", commit],
                           check=True, capture_output=True, text=True, timeout=120)
        else:
            # 分支/标签：浅克隆对应分支
            subprocess.run(prefix + ["clone", "--depth", "1", "--branch", commit,
                                     url, dest],
                           check=True, capture_output=True, text=True, timeout=timeout)
    else:
        # 默认分支：浅克隆
        subprocess.run(prefix + ["clone", "--depth", "1", url, dest],
                       check=True, capture_output=True, text=True, timeout=timeout)


def _tarball(url: str, ref: str, dest: str, prox: str | None,
             logger=None, timeout: int = 600):
    base = url.rstrip("/")
    # 分支/标签走 archive/refs/heads/<ref>.tar.gz；SHA 直接 archive/<sha>.tar.gz
    # （refs/heads/<sha> 不存在，GitHub 对 SHA 直接暴露 archive/<sha>.tar.gz）。
    archive_ref = ref if _looks_like_sha(ref) else f"refs/heads/{ref}"
    if prox:
        archive_url = f"{prox}{base}/archive/{archive_ref}.tar.gz"
    else:
        archive_url = f"{base}/archive/{archive_ref}.tar.gz"
    with tempfile.TemporaryDirectory() as td:
        tar_path = os.path.join(td, "repo.tar.gz")
        subprocess.run(["curl", "-fSL", archive_url, "-o", tar_path],
                       check=True, capture_output=True, text=True, timeout=timeout)
        import tarfile
        with tarfile.open(tar_path, "r:gz") as t:
            t.extractall(td)
        # 解压出 repo-ref 目录，移动到 dest
        extracted = None
        for name in os.listdir(td):
            if os.path.isdir(os.path.join(td, name)) and name != "repo.tar.gz":
                extracted = os.path.join(td, name)
                break
        if not extracted:
            raise RuntimeError("tarball 解压后未找到仓库目录")
        if os.path.exists(dest):
            shutil.rmtree(dest, ignore_errors=True)
        shutil.move(extracted, dest)


def download_repo(repo_spec: dict, dest: str, proxy_candidates: list,
                  logger=None, timeout: int = 600) -> bool:
    url = repo_spec["url"]
    commit = repo_spec.get("commit")
    if commit and commit.startswith("PINNED"):
        logger.warn("P1", "repo", f"commit 为占位哨兵，回退默认分支（不可复现）",
                    detail=repo_spec["id"])
        commit = None
    # 1) git 直连
    try:
        _git_clone(url, commit, dest, None, logger, timeout)
        return True
    except Exception as e:  # noqa: BLE001
        logger.warn("P1", "repo", f"git 直连失败 {repo_spec['id']}", detail=str(e)[:160])
    # 2) git 经代理
    for prox in proxy_candidates:
        try:
            _git_clone(url, commit, dest, prox, logger, timeout)
            return True
        except Exception as e:  # noqa: BLE001
            logger.warn("P1", "repo", f"git 代理失败 {repo_spec['id']}", detail=str(e)[:160])
    # 3) tarball 降级（直连 + 代理）。SHA 直接透传（_tarball 内部识别 SHA 用 archive/<sha>.tar.gz）；
    #    分支/标签透传；PINNED/None 回退 main。
    ref = commit if (commit and not commit.startswith("PINNED")) else "main"
    for prox in [None] + proxy_candidates:
        try:
            _tarball(url, ref, dest, prox, logger, timeout)
            return True
        except Exception as e:  # noqa: BLE001
            logger.warn("P1", "repo", f"tarball 失败 {repo_spec['id']}", detail=str(e)[:160])
    return False
