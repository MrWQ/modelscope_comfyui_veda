"""test_v2310.py — v2.3.10 自检集成测试。

覆盖：
  · _confirm_continue 的 6 个分支（dry_run / --yes / TTY-y / TTY-n / TTY-EOF / 非交互）
  · phase0_preflight 的可询问跳过场景：
      E103 无 GPU（非交互中止 / --yes 继续 / dry-run / TTY-y / TTY-n）
      E102 GPU 算力不足（非交互中止 / --yes / TTY-y / TTY-n）
      E601 外来 ComfyUI（非交互中止 / --force / --yes / TTY-y）
  · platform.compute_cap_ok 边界（7.5 / 8.0 / 8.6 / None）
  · diag._file_tree 跨平台归一化（.git/.cache/venv 跳过不递归）

直接 `python test_v2310.py` 运行（cwd=veda-setup）。无外部依赖，全 mock。
"""
import os
import sys
import tempfile
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from lib import phases, platform as real_platform, diag  # noqa: E402


# ---------------------------------------------------------------------------
# 假 logger（含全部被调用方法）
# ---------------------------------------------------------------------------
class L:
    def info(self, *a, **k): pass
    def warn(self, *a, **k): pass
    def ok(self, *a, **k): pass
    def error(self, *a, **k): pass
    def debug(self, *a, **k): pass


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
def main():
    test_confirm_branches()
    test_e103_no_gpu()
    test_e102_low_cap()
    test_e601_external()
    test_compute_cap_ok()
    test_file_tree()

    print("\n" + "=" * 60)
    print(f"PASS={len(PASS)}  FAIL={len(FAIL)}")
    if FAIL:
        print("FAILED:", FAIL)
        sys.exit(1)
    print("ALL_V2310_TESTS_OK")
    sys.exit(0)


if __name__ == "__main__":
    main()
