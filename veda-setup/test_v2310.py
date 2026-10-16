"""test_v2310.py — veda-setup 自检集成测试（覆盖 v2.3.10 → v2.3.16）。

覆盖：
  · _confirm_continue 的 6 个分支（dry_run / --yes / TTY-y / TTY-n / TTY-EOF / 非交互）
  · phase0_preflight 的可询问跳过场景：
      E103 无 GPU（非交互中止 / --yes 继续 / dry-run / TTY-y / TTY-n）
      E102 GPU 算力不足（非交互中止 / --yes / TTY-y / TTY-n）
      E601 外来 ComfyUI（非交互中止 / --force / --yes / TTY-y）
  · platform.compute_cap_ok 边界（7.5 / 8.0 / 8.6 / None）
  · diag._file_tree 跨平台归一化（.git/.cache/venv 跳过不递归）
  · v2.3.15：ensure_venv 健壮回退（复用/标准失败→--without-pip→系统 Python，置 venv_fallback）
  · v2.3.15：phase2_deps 在 venv_fallback 时跳过 requirements 安装（保护系统环境）
  · v2.3.15：_launch_precheck 识别非 CUDA 后端（AMD ROCm，torch.version.hip）
  · v2.3.16：pip 源可达性预检（可达/不可达）+ 流式安装（成功/无输出超时 kill）

直接 `python test_v2310.py` 运行（cwd=veda-setup）。无外部依赖，全 mock。
"""
import os
import sys
import json
import struct
import tempfile
import shutil
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from lib import phases, platform as real_platform, diag  # noqa: E402

# 让自检在任何宿主 Python 上都能跑（目标运行环境为 3.10-3.12，脚本有 E101
# 版本门禁；这里把 sys.version_info 钉到 3.12，避免开发机 3.13 误触发门禁）。
from collections import namedtuple  # noqa: E402

_VerInfo = namedtuple("version_info", "major minor micro releaselevel serial")  # noqa: E402
sys.version_info = _VerInfo(3, 12, 0, "final", 0)  # noqa: E402


# ---------------------------------------------------------------------------
# 假 logger（含全部被调用方法）
# ---------------------------------------------------------------------------
class L:
    def info(self, *a, **k): pass
    def warn(self, *a, **k): pass
    def ok(self, *a, **k): pass
    def error(self, *a, **k): pass
    def debug(self, *a, **k): pass


# 捕获日志事件（用于 final_summary 三态断言）
class CapL:
    def __init__(self):
        self.events = []
    def info(self, *a, **k): self.events.append(("info", k))
    def warn(self, *a, **k): self.events.append(("warn", k))
    def ok(self, *a, **k): self.events.append(("ok", k))
    def error(self, *a, **k): self.events.append(("error", k))
    def debug(self, *a, **k): pass


# 带 record_asset/is_asset_done 的假 State（供下载测试）
class FakeState2:
    def __init__(self):
        self.data = {"assets": {}}
    def load(self): pass
    def meta_changed(self, *a, **k): return False
    def set_meta(self, *a, **k): pass
    def is_asset_done(self, aid, fp):
        a = self.data.get("assets", {}).get(aid)
        return bool(a and a.get("status") == "ok" and a.get("fingerprint") == fp)
    def record_asset(self, aid, status, size=None, fingerprint=None, source=None):
        self.data.setdefault("assets", {})[aid] = {
            "status": status, "size": size, "fingerprint": fingerprint,
            "source": source}


# ---------------------------------------------------------------------------
# 假 Args
# ---------------------------------------------------------------------------
class Args:
    def __init__(self, **kw):
        self.dry_run = False
        self.yes = False
        self.force = False
        self.profile = "auto"
        self.accept_third_party = False
        self.skip_models = True
        self.mode = "auto"
        self.only = None
        self.source_priority = None
        self.no_launch = False
        self.listen = False
        self.__dict__.update(kw)


# ---------------------------------------------------------------------------
# 假 State
# ---------------------------------------------------------------------------
class FakeState:
    def __init__(self):
        self.data = {}
    def load(self): pass
    def meta_changed(self, *a, **k): return False
    def set_meta(self, *a, **k): pass


# ---------------------------------------------------------------------------
# 假 TTY（isatty=True，readline 返回答案）
# ---------------------------------------------------------------------------
class FakeTTY:
    def __init__(self, answer):
        self._buf = answer + "\n"
    def isatty(self):
        return True
    def readline(self):
        return self._buf


# ---------------------------------------------------------------------------
# phase0 运行上下文构造
# ---------------------------------------------------------------------------
PROFILES = {
    "standard": {"asset_ids": []},
    "lowvram": {"require_third_party": True, "asset_ids": []},
    "full": {"asset_ids": []},
    "min": {"asset_ids": []},
}

# 正常 GPU（SM89=RTX40 系，VRAM 48G → full 档，算力达标不触发 E102）
OK_GPU = {"name": "RTX 4090", "vram_gb": 48.0, "compute_cap": "8.9"}


def mkctx(args_override=None, gpu=None, comfyui_present=False, tty_answer=None):
    ctx = phases.Ctx()
    ctx.root = tempfile.mkdtemp(prefix="veda_test_")
    ctx.comfyui_dir = os.path.join(ctx.root, "ComfyUI")
    ctx.manifest = {"assets": []}
    ctx.manifest_sha1 = "deadbeef"
    ctx.manifest_profiles = PROFILES
    ctx.state = FakeState()
    ctx.logger = L()
    ctx.args = Args(**(args_override or {}))

    if comfyui_present:
        os.makedirs(ctx.comfyui_dir, exist_ok=True)
        # 外来 ComfyUI：存在目录但无 .git；本测试用 comfyui_present 表示"存在外来"
        # （即 isdir=True 且 .git 不存在）
    else:
        # 不存在目录（isdir=False）→ 不触发 E601
        ctx.comfyui_dir = os.path.join(ctx.root, "no_such_comfyui")

    # mock platform 函数
    real_platform.detect_platform = lambda: {"os": "linux", "is_modelscope": False}
    real_platform.detect_gpu = lambda: gpu
    real_platform.network_probe = lambda *a, **k: {"modelscope": True}
    real_platform.venv_python = lambda root: os.path.join(root, "venv", "bin", "python")
    real_platform.disk_free = lambda root: 10 * 1024 ** 3

    if tty_answer is not None:
        sys.stdin = FakeTTY(tty_answer)
    return ctx


def cleanup(ctx):
    try:
        shutil.rmtree(ctx.root, ignore_errors=True)
    finally:
        sys.stdin = sys.__stdin__


# ---------------------------------------------------------------------------
# 7) phase8_launch 崩溃/超时/就绪（mock subprocess + 探针）
# ---------------------------------------------------------------------------
class FakeProc:
    def __init__(self, rc):
        self._rc = rc
    def poll(self):
        return self._rc


def make_p8_ctx():
    ctx = phases.Ctx()
    ctx.root = tempfile.mkdtemp(prefix="veda_p8_")
    ctx.comfyui_dir = os.path.join(ctx.root, "ComfyUI")
    ctx.venv_python = sys.executable  # 实际不执行（Popen 被 mock）
    ctx.pip_cache = os.path.join(ctx.root, "pip_cache")  # 避免流式安装取 None
    ctx.manifest = {"assets": []}
    ctx.manifest_sha1 = "x"
    ctx.manifest_profiles = PROFILES
    ctx.state = FakeState()
    ctx.logger = L()
    ctx.args = Args(no_launch=False, listen=False)
    return ctx


def test_p8_launch():
    print("\n[7] phase8_launch 崩溃/超时/就绪")
    import lib.phases as P

    # 就绪：探针 True → launch_ready=True，不抛错
    c = make_p8_ctx()
    with mock.patch.object(P, "_poll_object_info", return_value=True), \
         mock.patch.object(P, "_port_ready", return_value=False), \
         mock.patch.object(P, "_launch_precheck", return_value=(True, "")), \
         mock.patch.object(P, "subprocess") as msp:
        msp.Popen.return_value = FakeProc(None)
        phases.phase8_launch(c)
    check("p8_ready_ok", c.launch_ready is True)
    cleanup(c)

    # 崩溃：探针 False 且进程已退出 → 抛 PhaseError(701)
    c = make_p8_ctx()
    with mock.patch.object(P, "_poll_object_info", return_value=False), \
         mock.patch.object(P, "_port_ready", return_value=False), \
         mock.patch.object(P, "_launch_precheck", return_value=(True, "")), \
         mock.patch.object(P, "subprocess") as msp:
        msp.Popen.return_value = FakeProc(1)
        try:
            phases.phase8_launch(c)
            check("p8_crash_raises", False)
        except phases.PhaseError as e:
            check("p8_crash_raises", e.code == 701)
    cleanup(c)

    # 存活但超时：探针 False 且进程仍运行 → 不抛错，launch_ready=False（诚实）
    c = make_p8_ctx()
    with mock.patch.object(P, "_poll_object_info", return_value=False), \
         mock.patch.object(P, "_port_ready", return_value=False), \
         mock.patch.object(P, "_launch_precheck", return_value=(True, "")), \
         mock.patch.object(P, "subprocess") as msp:
        msp.Popen.return_value = FakeProc(None)
        phases.phase8_launch(c)
    check("p8_alive_timeout_noraise", c.launch_ready is False)
    cleanup(c)

    # final_summary 在 launch_ready=False 时诚实提示、不抛错
    c = make_p8_ctx()
    c.launch_ready = False
    phases.final_summary(c, "launch-only")
    check("final_summary_notready_ok", True)
    cleanup(c)


# ---------------------------------------------------------------------------
# 11) v2.3.14：P8 启动前置硬门禁（环境不满足不 spawn 必崩进程）
# ---------------------------------------------------------------------------
def test_p8_launch_gate():
    print("\n[11] phase8_launch 启动前置门禁（v2.3.14）")
    import lib.phases as P
    import subprocess as _sp

    # 环境不满足 + strict=True（--mode launch / --only p8）：直接抛 E701，绝不 spawn
    c = make_p8_ctx()
    with mock.patch.object(P, "_launch_precheck", return_value=(False, "未检测到 NVIDIA GPU，Veda 需 GPU 才能运行")), \
         mock.patch.object(P, "subprocess") as msp:
        try:
            phases.phase8_launch(c, strict=True)
            check("p8_gate_strict_raises", False)
        except phases.PhaseError as e:
            check("p8_gate_strict_raises", e.code == 701)
    check("p8_gate_strict_no_spawn", msp.Popen.call_count == 0)

    # 环境不满足 + strict=False（auto/install 顺带启动）：跳过，不崩溃，不 spawn
    c = make_p8_ctx()
    with mock.patch.object(P, "_launch_precheck", return_value=(False, "torch 运行时不兼容：torch 版本过低")), \
         mock.patch.object(P, "subprocess") as msp:
        phases.phase8_launch(c, strict=False)
    check("p8_gate_besteffort_skip", c.launch_ready is None
          and c.launch_skipped_reason is not None)
    check("p8_gate_besteffort_no_spawn", msp.Popen.call_count == 0)
    check("p8_gate_besteffort_reason", "torch" in (c.launch_skipped_reason or ""))

    # 环境满足：正常 spawn（Popen 被调用）
    c = make_p8_ctx()
    with mock.patch.object(P, "_launch_precheck", return_value=(True, "")), \
         mock.patch.object(P, "_poll_object_info", return_value=True), \
         mock.patch.object(P, "_port_ready", return_value=False), \
         mock.patch.object(P, "subprocess") as msp:
        msp.Popen.return_value = FakeProc(None)
        phases.phase8_launch(c, strict=False)
    check("p8_gate_ok_spawn", msp.Popen.call_count == 1 and c.launch_ready is True)


# ---------------------------------------------------------------------------
# 12) v2.3.14：_launch_precheck 四种环境判定
# ---------------------------------------------------------------------------
def test_launch_precheck():
    print("\n[12] _launch_precheck 环境判定（v2.3.14）")
    import subprocess as _sp

    def fake_run(stdout, returncode=0, stderr=""):
        return _sp.CompletedProcess(args=[], returncode=returncode,
                                    stdout=stdout, stderr=stderr)

    c = make_p8_ctx()

    # venv 缺失
    c.venv_python = os.path.join(c.root, "no_such_venv", "python")
    ok, reason = phases._launch_precheck(c)
    check("precheck_venv_missing", ok is False and "venv" in reason)

    # venv 存在 + torch 不兼容（缺 custom_op）
    c.venv_python = sys.executable
    with mock.patch.object(phases, "_torch_compat_ok", return_value=(False, "torch too old: missing custom_op")), \
         mock.patch.object(phases, "_run_venv", return_value=fake_run("nocuda")):
        ok, reason = phases._launch_precheck(c)
    check("precheck_torch_incompat", ok is False and "torch" in reason)

    # venv 存在 + torch OK 但无 GPU
    with mock.patch.object(phases, "_torch_compat_ok", return_value=(True, "2.1.0")), \
         mock.patch.object(phases, "_run_venv", return_value=fake_run("nocuda")):
        ok, reason = phases._launch_precheck(c)
    check("precheck_no_gpu", ok is False and "GPU" in reason)

    # venv 存在 + torch OK + 有 GPU
    with mock.patch.object(phases, "_torch_compat_ok", return_value=(True, "2.1.0")), \
         mock.patch.object(phases, "_run_venv", return_value=fake_run("cuda")):
        ok, reason = phases._launch_precheck(c)
    check("precheck_ok", ok is True and reason == "")

    # v2.3.15：非 CUDA 后端（AMD ROCm）——torch.version.hip 存在
    c2 = make_p8_ctx()
    with mock.patch.object(phases, "_torch_compat_ok", return_value=(True, "2.12.0")), \
         mock.patch.object(phases, "_run_venv", return_value=fake_run("hip")):
        ok, reason = phases._launch_precheck(c2)
    check("precheck_rocm", ok is False and "ROCm" in reason)
    cleanup(c2)


PASS = []
FAIL = []


def check(name, cond):
    (PASS if cond else FAIL).append(name)
    print(("  ok  " if cond else " FAIL ") + name)


# ---------------------------------------------------------------------------
# 1) _confirm_continue 6 分支
# ---------------------------------------------------------------------------
def test_confirm_branches():
    print("\n[1] _confirm_continue 分支")

    # dry_run
    c = mkctx({"dry_run": True})
    check("confirm_dryrun_true", phases._confirm_continue(c, "x") is True)
    cleanup(c)

    # --yes
    c = mkctx({"yes": True})
    check("confirm_yes_true", phases._confirm_continue(c, "x") is True)
    cleanup(c)

    # 非交互（真实 stdin 非 tty）
    c = mkctx({})
    real_stdin = sys.stdin
    try:
        sys.stdin = open(os.devnull)  # 非 tty
        check("confirm_noninteractive_false", phases._confirm_continue(c, "x") is False)
    finally:
        sys.stdin = real_stdin
    cleanup(c)

    # TTY + y
    c = mkctx({})
    sys.stdin = FakeTTY("y")
    check("confirm_tty_y_true", phases._confirm_continue(c, "x") is True)
    cleanup(c)

    # TTY + n
    c = mkctx({})
    sys.stdin = FakeTTY("n")
    check("confirm_tty_n_false", phases._confirm_continue(c, "x") is False)
    cleanup(c)

    # TTY + EOF（空）
    c = mkctx({})
    sys.stdin = FakeTTY("")
    check("confirm_tty_eof_false", phases._confirm_continue(c, "x") is False)
    cleanup(c)


# ---------------------------------------------------------------------------
# 2) E103 无 GPU
# ---------------------------------------------------------------------------
def test_e103_no_gpu():
    print("\n[2] E103 无 GPU 可询问跳过")

    # 非交互无 --yes → 中止
    c = mkctx({}, gpu=None)
    real_stdin = sys.stdin
    try:
        sys.stdin = open(os.devnull)
        phases.phase0_preflight(c)
    finally:
        sys.stdin = real_stdin
    check("e103_nogpu_abort", c.user_aborted is True)
    cleanup(c)

    # --yes → 继续，用 standard
    c = mkctx({"yes": True}, gpu=None)
    phases.phase0_preflight(c)
    check("e103_nogpu_yes_continue", c.user_aborted is False and c.profile_name == "standard")
    cleanup(c)

    # dry_run → 继续
    c = mkctx({"dry_run": True}, gpu=None)
    phases.phase0_preflight(c)
    check("e103_nogpu_dryrun_continue", c.user_aborted is False and c.profile_name == "standard")
    cleanup(c)

    # TTY y → 继续
    c = mkctx({}, gpu=None, tty_answer="y")
    phases.phase0_preflight(c)
    check("e103_nogpu_tty_y_continue", c.user_aborted is False and c.profile_name == "standard")
    cleanup(c)

    # TTY n → 中止
    c = mkctx({}, gpu=None, tty_answer="n")
    phases.phase0_preflight(c)
    check("e103_nogpu_tty_n_abort", c.user_aborted is True)
    cleanup(c)


# ---------------------------------------------------------------------------
# 3) E102 GPU 算力不足（compute_cap < SM80）
# ---------------------------------------------------------------------------
def test_e102_low_cap():
    print("\n[3] E102 GPU 算力不足可询问跳过")

    low_gpu = {"name": "GT 1030", "vram_gb": 48.0, "compute_cap": "7.5"}

    # 非交互无 --yes → 中止
    c = mkctx({}, gpu=low_gpu)
    real_stdin = sys.stdin
    try:
        sys.stdin = open(os.devnull)
        phases.phase0_preflight(c)
    finally:
        sys.stdin = real_stdin
    check("e102_lowcap_abort", c.user_aborted is True)
    cleanup(c)

    # --yes → 继续（full 档）
    c = mkctx({"yes": True}, gpu=low_gpu)
    phases.phase0_preflight(c)
    check("e102_lowcap_yes_continue", c.user_aborted is False and c.profile_name == "full")
    cleanup(c)

    # TTY y → 继续
    c = mkctx({}, gpu=low_gpu, tty_answer="y")
    phases.phase0_preflight(c)
    check("e102_lowcap_tty_y_continue", c.user_aborted is False and c.profile_name == "full")
    cleanup(c)

    # TTY n → 中止
    c = mkctx({}, gpu=low_gpu, tty_answer="n")
    phases.phase0_preflight(c)
    check("e102_lowcap_tty_n_abort", c.user_aborted is True)
    cleanup(c)

    # 正常算力（8.9）+ 48G VRAM → 不触发 confirm，full 档
    c = mkctx({}, gpu=OK_GPU)
    phases.phase0_preflight(c)
    check("e102_normal_cap_no_abort", c.user_aborted is False and c.profile_name == "full")
    cleanup(c)


# ---------------------------------------------------------------------------
# 4) E601 外来 ComfyUI
# ---------------------------------------------------------------------------
def test_e601_external():
    print("\n[4] E601 外来 ComfyUI 可询问跳过")

    # 非交互无 --yes → 中止（用正常 GPU 隔离 E103，专测 E601）
    c = mkctx({}, gpu=OK_GPU, comfyui_present=True)
    real_stdin = sys.stdin
    try:
        sys.stdin = open(os.devnull)
        phases.phase0_preflight(c)
    finally:
        sys.stdin = real_stdin
    check("e601_external_abort", c.user_aborted is True)
    cleanup(c)

    # --force → 覆盖继续
    c = mkctx({"force": True}, gpu=OK_GPU, comfyui_present=True)
    phases.phase0_preflight(c)
    check("e601_force_continue", c.user_aborted is False)
    cleanup(c)

    # --yes → 继续
    c = mkctx({"yes": True}, gpu=OK_GPU, comfyui_present=True)
    phases.phase0_preflight(c)
    check("e601_yes_continue", c.user_aborted is False)
    cleanup(c)

    # TTY y → 继续
    c = mkctx({}, gpu=OK_GPU, comfyui_present=True, tty_answer="y")
    phases.phase0_preflight(c)
    check("e601_tty_y_continue", c.user_aborted is False)
    cleanup(c)


# ---------------------------------------------------------------------------
# 5) compute_cap_ok 边界
# ---------------------------------------------------------------------------
def test_compute_cap_ok():
    print("\n[5] platform.compute_cap_ok 边界")
    check("cap_7.5_false", real_platform.compute_cap_ok({"compute_cap": "7.5"}) is False)
    check("cap_8.0_true", real_platform.compute_cap_ok({"compute_cap": "8.0"}) is True)
    check("cap_8.6_true", real_platform.compute_cap_ok({"compute_cap": "8.6"}) is True)
    check("cap_none_false", real_platform.compute_cap_ok(None) is False)
    check("cap_missing_false", real_platform.compute_cap_ok({}) is False)


# ---------------------------------------------------------------------------
# 6) _file_tree 跨平台归一化
# ---------------------------------------------------------------------------
def test_file_tree():
    print("\n[6] diag._file_tree 跨平台归一化")
    root = tempfile.mkdtemp(prefix="veda_ft_")
    try:
        # 普通文件
        with open(os.path.join(root, "readme.txt"), "w") as f:
            f.write("x")
        # .git 目录（应跳过递归）
        os.makedirs(os.path.join(root, ".git", "objects"))
        with open(os.path.join(root, ".git", "objects", "big"), "w") as f:
            f.write("y" * 1000)
        # venv 目录（应跳过递归）
        os.makedirs(os.path.join(root, "venv", "lib"))
        with open(os.path.join(root, "venv", "lib", "huge"), "w") as f:
            f.write("z" * 1000)
        # .cache 目录（应跳过递归）
        os.makedirs(os.path.join(root, ".cache"))
        with open(os.path.join(root, ".cache", "model"), "w") as f:
            f.write("w" * 1000)

        out = diag._file_tree(root)
        # .git/objects/big 不应被列出（被 skipped 阻断）
        check("ft_git_not_recursed", "objects/big" not in out and ".git" in out)
        # venv 递归被阻断：huge 文件不应出现；venv 目录应在 skipped 行出现（跨平台分隔符）
        check("ft_venv_not_recursed", "huge" not in out and "venv" in out)
        check("ft_cache_not_recursed", "model" not in out and ".cache" in out)
        check("ft_normal_listed", "readme.txt" in out)
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ---------------------------------------------------------------------------
# 8) safetensors 数据段完整性校验（v2.3.12：截断但头部合法应判失败）
# ---------------------------------------------------------------------------
def make_st(data_len: int) -> bytes:
    """构造一个合法的单张量 safetensors（header + data_len 字节数据段）。"""
    header = {"t": {"dtype": "F32", "shape": [data_len // 4],
                    "data_offsets": [0, data_len]}}
    raw = json.dumps(header).encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw + b"\x00" * data_len


def test_safetensors_integrity():
    print("\n[8] safetensors 数据段完整性")
    import lib.assets as A
    d = tempfile.mkdtemp(prefix="veda_st_")
    try:
        good = os.path.join(d, "good.safetensors")
        content = make_st(8)
        with open(good, "wb") as f:
            f.write(content)
        check("st_good_true", A.verify_safetensors_header(good) is True)
        # 头部合法但数据段被截断（模拟下载中断）：尾部少 4 字节
        trunc = os.path.join(d, "trunc.safetensors")
        with open(trunc, "wb") as f:
            f.write(content[:len(content) - 4])
        check("st_trunc_false", A.verify_safetensors_header(trunc) is False)
        # 头部长度非法
        bad = os.path.join(d, "bad.safetensors")
        with open(bad, "wb") as f:
            f.write(b"\xff\xff\xff\xff\xff\xff\xff\xff" + b"{}")
        check("st_badhdr_false", A.verify_safetensors_header(bad) is False)
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# 9) download_model_asset：恢复完整文件 + 清理残次文件（v2.3.12）
# ---------------------------------------------------------------------------
def test_download_recover_and_cleanup():
    print("\n[9] download_model_asset 恢复/清理")
    import lib.assets as A
    root = tempfile.mkdtemp(prefix="veda_dl_")
    try:
        asset = {"id": "t1",
                 "target": "ComfyUI/models/veda/t1.safetensors",
                 "min_bytes": 0,
                 "sources": [{"type": "hf", "repo": "x", "file": "t1.safetensors"},
                             {"type": "hf", "repo": "x", "file": "t1.safetensors"}]}
        env = {"is_modelscope": False, "gh_proxy_ok": False,
               "modelscope_cache": None, "hf_cache": None}
        vdir = os.path.join(root, "ComfyUI", "models", "veda")
        os.makedirs(vdir, exist_ok=True)
        target = os.path.join(vdir, "t1.safetensors")

        # 9a 恢复：完整文件已存在但 state 未记录 → 直接复用，不调用下载
        with open(target, "wb") as f:
            f.write(make_st(8))
        state = FakeState2()
        logger = L()
        ok, src = A.download_model_asset(asset, root, [], env, True, logger, state)
        check("dl_recover_ok", ok is True and src == "recovered")
        check("dl_recover_recorded", state.data["assets"]["t1"]["status"] == "ok")

        # 9b 清理：残次文件存在 → 重试前被删除；第二次源才成功落盘
        os.remove(target)
        with open(target, "wb") as f:   # 残次（过小，头部非法）
            f.write(b"\x00" * 6)
        state = FakeState2()
        logger = L()
        before_exists = []
        def fake_hf(repo, include, local_dir, endpoint, cache_dir, logger=None, timeout=1800):
            t = os.path.join(local_dir, include)
            before_exists.append(os.path.exists(t))  # 进入下载前 target 是否已被清理
            if len(before_exists) == 1:
                raise RuntimeError("first source failed (simulated)")
            # 第二次：写入完整合法文件
            with open(t, "wb") as f:
                f.write(make_st(8))
        with mock.patch.object(A, "_dl_hf", side_effect=fake_hf):
            ok2, src2 = A.download_model_asset(asset, root, asset["sources"], env, True, logger, state)
        check("dl_cleanup_then_ok", ok2 is True and src2 == "hf")
        # 第二次源进入下载前，残次 target 应已被清理（before_exists[1] == False）
        check("dl_partial_cleaned", len(before_exists) == 2 and before_exists[1] is False)
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ---------------------------------------------------------------------------
# 10) final_summary 三态：未启动不得谎报可访问 URL（v2.3.12）
# ---------------------------------------------------------------------------
def test_final_summary_states():
    print("\n[10] final_summary 三态（未启动不谎报 URL）")

    # ready=None（--no-launch / 仅安装环节）：不得带 access 字段
    c = make_p8_ctx(); c.launch_ready = None; cl = CapL(); c.logger = cl
    phases.final_summary(c, "install：安装+启动")
    ok_events = [e for e in cl.events if e[0] == "ok"]
    has_access = any(isinstance(e[1].get("detail"), dict) and "access" in e[1]["detail"]
                     for e in ok_events)
    check("fs_none_no_access", has_access is False)

    # ready=True：带 access 字段（已就绪，可访问）
    c = make_p8_ctx(); c.launch_ready = True; cl = CapL(); c.logger = cl
    phases.final_summary(c, "launch-only")
    ok_events = [e for e in cl.events if e[0] == "ok"]
    has_access = any(isinstance(e[1].get("detail"), dict) and "access" in e[1]["detail"]
                     for e in ok_events)
    check("fs_true_has_access", has_access is True)

    # ready=False：warn（诚实未就绪），不抛错
    c = make_p8_ctx(); c.launch_ready = False; cl = CapL(); c.logger = cl
    phases.final_summary(c, "launch-only")
    warn_events = [e for e in cl.events if e[0] == "warn"]
    check("fs_false_warn", len(warn_events) >= 1)

    # v2.3.14：环境不满足跳过启动（launch_skipped_reason 已设置）→ warn + 原因，不谎报可访问
    c = make_p8_ctx(); c.launch_ready = None
    c.launch_skipped_reason = "未检测到 NVIDIA GPU，Veda 需 GPU 才能运行"
    cl = CapL(); c.logger = cl
    phases.final_summary(c, "install：安装+启动")
    warn_events = [e for e in cl.events if e[0] == "warn"]
    has_access = any(isinstance(e[1].get("detail"), dict) and "access" in e[1]["detail"]
                     for e in warn_events)
    check("fs_skip_warn", len(warn_events) >= 1 and has_access is False)


# ---------------------------------------------------------------------------
# v2.3.13：torch API 兼容性探针（comfy_kitchen 需 torch.library.custom_op / torch>=2.0）
def test_torch_compat():
    import subprocess as _sp

    def fake_run(stdout, returncode=0, stderr=""):
        return _sp.CompletedProcess(args=[], returncode=returncode,
                                    stdout=stdout, stderr=stderr)

    with mock.patch.object(phases, "_run_venv",
                           return_value=fake_run("2.1.0\ncustom_op\n")):
        ok, detail = phases._torch_compat_ok("fakepython")
        check("torch_compat_ok", ok is True and detail == "2.1.0")

    with mock.patch.object(phases, "_run_venv",
                           return_value=fake_run("1.13.1\nno_custom_op\n")):
        ok, detail = phases._torch_compat_ok("fakepython")
        check("torch_compat_too_old", ok is False and "torch>=2.0" in detail)

    with mock.patch.object(phases, "_run_venv",
                           return_value=fake_run("", returncode=1,
                                                stderr="ImportError: no torch")):
        ok, detail = phases._torch_compat_ok("fakepython")
        check("torch_compat_import_fail", ok is False and "torch 无法导入" in detail)


# ---------------------------------------------------------------------------
# 13) v2.3.15：ensure_venv 健壮回退（复用 / 标准失败→--without-pip→系统 Python）
# ---------------------------------------------------------------------------
def test_ensure_venv_fallback():
    print("\n[13] platform.ensure_venv 回退链（v2.3.15）")
    import subprocess as _sp

    def fake_run_factory(root, scenario):
        def fake_run(cmd, *a, **k):
            if "-m" in cmd and "venv" in cmd:
                if scenario == "reuse":
                    return _sp.CompletedProcess(cmd, 0, "", "")
                if scenario == "without_pip_ok":
                    if "--without-pip" in cmd:
                        vpy = real_platform.venv_python(root)
                        os.makedirs(os.path.dirname(vpy), exist_ok=True)
                        open(vpy, "w").close()
                        return _sp.CompletedProcess(cmd, 0, "", "")
                    return _sp.CompletedProcess(cmd, 1, "", "ensurepip not available")
                # all_fail：两种尝试都失败
                return _sp.CompletedProcess(cmd, 1, "", "ensurepip not available")
            if "-c" in cmd:
                return _sp.CompletedProcess(cmd, 0, "3.12.0", "")
            return _sp.CompletedProcess(cmd, 0, "", "")
        return fake_run

    # 场景 A：已有可用 venv → 复用，venv_fallback=False（不触发 venv 创建命令）
    root_a = tempfile.mkdtemp(prefix="veda_vnv_a_")
    c_a = phases.Ctx()
    vpy_a = real_platform.venv_python(root_a)
    os.makedirs(os.path.dirname(vpy_a), exist_ok=True)
    open(vpy_a, "w").close()
    rc0 = _sp.CompletedProcess(args=[], returncode=0, stdout="3.12.0", stderr="")
    with mock.patch.object(real_platform.subprocess, "run") as mrun:
        mrun.return_value = rc0
        got = real_platform.ensure_venv(root_a, True, L(), c_a)
    check("ensure_venv_reuse", got == vpy_a and c_a.venv_fallback is False)
    venv_create_calls = [cl for cl in mrun.call_args_list
                         if cl.args and "-m" in cl.args[0] and "venv" in cl.args[0]]
    check("ensure_venv_reuse_no_create", len(venv_create_calls) == 0)
    shutil.rmtree(root_a, ignore_errors=True)

    # 场景 B：标准 venv 失败 → --without-pip 成功（仍视为 fallback）
    root_b = tempfile.mkdtemp(prefix="veda_vnv_b_")
    c_b = phases.Ctx()
    with mock.patch.object(real_platform.subprocess, "run",
                          side_effect=fake_run_factory(root_b, "without_pip_ok")):
        got = real_platform.ensure_venv(root_b, True, L(), c_b)
    check("ensure_venv_without_pip", got == real_platform.venv_python(root_b)
          and c_b.venv_fallback is True)
    shutil.rmtree(root_b, ignore_errors=True)

    # 场景 C：全部失败 → 回退系统 Python，venv_fallback=True，不抛错
    root_c = tempfile.mkdtemp(prefix="veda_vnv_c_")
    c_c = phases.Ctx()
    with mock.patch.object(real_platform.subprocess, "run",
                          side_effect=fake_run_factory(root_c, "all_fail")):
        got = real_platform.ensure_venv(root_c, True, L(), c_c)
    check("ensure_venv_fallback_syspy", got == sys.executable
          and c_c.venv_fallback is True)
    shutil.rmtree(root_c, ignore_errors=True)


# ---------------------------------------------------------------------------
# 14) v2.3.15：phase2_deps 在 venv_fallback 时跳过 requirements 安装
# ---------------------------------------------------------------------------
def test_phase2_venv_fallback():
    print("\n[14] phase2_deps venv 回退跳过 requirements（v2.3.15）")
    import subprocess as _sp
    import lib.phases as P

    rc0 = _sp.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    # 回退场景：ensure_venv 返回系统 python 且置 venv_fallback=True
    c = make_p8_ctx()
    os.makedirs(c.comfyui_dir, exist_ok=True)
    with open(os.path.join(c.comfyui_dir, "requirements.txt"), "w") as f:
        f.write("some-pkg\n")
    with mock.patch.object(P.platform, "ensure_venv",
                           side_effect=lambda *a, **k: (k.get("ctx") or a[3]).__setattr__("venv_fallback", True) or sys.executable), \
         mock.patch.object(P, "subprocess") as msp:
        msp.run.return_value = rc0
        P.phase2_deps(c)
    check("phase2_fallback_flag", c.venv_fallback is True)
    check("phase2_fallback_py", c.venv_python == sys.executable)
    pip_req = [cl for cl in msp.run.call_args_list
               if cl.args and "pip" in cl.args[0] and "-r" in cl.args[0]]
    check("phase2_fallback_skip_req", len(pip_req) == 0)
    cleanup(c)

    # 正常场景：venv 就绪 → 预检通过 → 流式 pip install -r（经 Popen，非 subprocess.run）
    c2 = make_p8_ctx()
    os.makedirs(c2.comfyui_dir, exist_ok=True)
    with open(os.path.join(c2.comfyui_dir, "requirements.txt"), "w") as f:
        f.write("some-pkg\n")
    vpy2 = (os.path.join(c2.root, "venv", "Scripts", "python.exe")
            if os.name == "nt" else os.path.join(c2.root, "venv", "bin", "python"))

    class _StreamProc:
        def __init__(self):
            self.stdout = iter(["Downloading some-pkg (10 kB)",
                                "Installing some-pkg",
                                "Successfully installed some-pkg"])
        def wait(self, timeout=None):
            return 0
        def kill(self):
            pass

    with mock.patch.object(P.platform, "ensure_venv", return_value=vpy2), \
         mock.patch.object(P, "_pip_index_reachable", return_value=(True, "pypi.org")), \
         mock.patch.object(P, "subprocess") as msp2:
        msp2.run.return_value = rc0  # 供 Windows triton 探测分支使用
        msp2.Popen.return_value = _StreamProc()
        P.phase2_deps(c2)
    check("phase2_normal_no_fallback", c2.venv_fallback is False)
    pip_calls = [cl for cl in msp2.Popen.call_args_list
                 if cl.args and isinstance(cl.args[0], list)
                 and "pip" in cl.args[0] and "-r" in cl.args[0]]
    check("phase2_normal_req_install", len(pip_calls) == 1)
    cleanup(c2)


# ---------------------------------------------------------------------------
# 15) v2.3.16：pip 源可达性预检 + 流式安装（避免 P2 静默卡死）
# ---------------------------------------------------------------------------
def test_pip_streaming():
    print("\n[15] pip 源预检 + 流式安装（v2.3.16）")
    import lib.phases as P
    import urllib.request as _ur

    # _pip_index_reachable：索引可达
    class _CtxOk:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self, n=1): return b"x"
    with mock.patch.object(_ur, "urlopen", return_value=_CtxOk()):
        ok, msg = P._pip_index_reachable(L())
    check("pip_reach_ok", ok is True and "pypi.org" in msg)

    # _pip_index_reachable：索引不可达（urlopen 抛异常）
    with mock.patch.object(_ur, "urlopen", side_effect=Exception("conn timeout")):
        ok, msg = P._pip_index_reachable(L())
    check("pip_reach_fail", ok is False and "不可达" in msg)

    # _pip_install_streaming：成功（rc=0，逐行读取并收集 tail）
    c = make_p8_ctx()
    class _GoodProc:
        def __init__(self):
            self.stdout = iter(["Downloading torch (800 MB)",
                                "Installing torch",
                                "Successfully installed torch"])
        def wait(self, timeout=None): return 0
        def kill(self): pass
    with mock.patch.object(P, "subprocess") as msp:
        msp.Popen.return_value = _GoodProc()
        rc, tail = P._pip_install_streaming(c, "req.txt", L())
    check("pip_stream_rc0", rc == 0)
    check("pip_stream_tail", "Successfully installed torch" in tail)
    cleanup(c)

    # _pip_install_streaming：无输出超时 → kill 并返回 rc=1（1s 窗口快速验证）
    c2 = make_p8_ctx()
    class _BlockIter:
        def __iter__(self): return self
        def __next__(self):
            import time as _t
            while True:
                _t.sleep(0.3)  # 永不产出下一行，阻塞读取（模拟源黑洞）
    class _BlockProc:
        def __init__(self):
            self.stdout = _BlockIter()
        def wait(self, timeout=None): return 0
        def kill(self): pass
    with mock.patch.object(P, "subprocess") as msp2:
        msp2.Popen.return_value = _BlockProc()
        rc2, tail2 = P._pip_install_streaming(c2, "req.txt", L(),
                                              overall_timeout=5, no_output_timeout=1)
    check("pip_stream_noinput_kill", rc2 == 1)
    cleanup(c2)


# ---------------------------------------------------------------------------
def main():
    test_confirm_branches()
    test_e103_no_gpu()
    test_e102_low_cap()
    test_e601_external()
    test_compute_cap_ok()
    test_file_tree()
    test_p8_launch()
    test_p8_launch_gate()
    test_launch_precheck()
    test_torch_compat()
    test_safetensors_integrity()
    test_download_recover_and_cleanup()
    test_final_summary_states()
    test_ensure_venv_fallback()
    test_phase2_venv_fallback()
    test_pip_streaming()

    print("\n" + "=" * 60)
    print(f"PASS={len(PASS)}  FAIL={len(FAIL)}")
    if FAIL:
        print("FAILED:", FAIL)
        sys.exit(1)
    print("ALL_V2310_TESTS_OK")
    sys.exit(0)


if __name__ == "__main__":
    main()
