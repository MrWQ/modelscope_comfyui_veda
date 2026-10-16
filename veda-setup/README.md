# Veda-on-ComfyUI 一键搭建脚本

在魔搭社区（ModelScope）免费 Notebook 等环境**一键部署 Veda-on-ComfyUI**（MiniMax-H3 视频扩散模型的官方加速节点，学习型稀疏注意力，约 3× 提速）。

> 设计文档：`docs/Veda一键脚本设计文档.md`（v2.3.16）。本脚本严格按设计文档实现：多平台、持久化免重装、**安装/启动自动分流**、全自动、优先魔搭下载能力、结构化日志 + 诊断包；v2.3.7 含首次魔搭实例实测反馈修复（模型下载落点 / hf 兜底源链 / 阶段级快速跳过 / `--only` 单环节执行），v2.3.8 含全量自检修正（P8 已运行守护与会话脱离、tarball 幂等标记、通用源链 HF 直连兜底等），v2.3.9 含无 GPU 交互继续，v2.3.10 将同类"可询问跳过"扩展到 E102（算力不足）与 E601（外来 ComfyUI），并修正 `compute_cap_ok` 死代码、诊断包 Windows 分隔符等自检问题；v2.3.11 含 P8 启动诚实化修复（启动崩溃/超时不再谎报成功：崩溃抛 E701 + 就地打印 `run/comfyui.log` 尾部，超时诚实标记未就绪；launch 预检补 CUDA 可用性早期信号）；v2.3.12 含同类"谎报/不完整"缺陷修复（`final_summary` 三态——未启动不再谎报可访问 URL；`verify_safetensors_header` 数据段完整性防"下载不完整却判定成功"；`download_model_asset` 回收完整文件 + 清理半截 target；损坏 `.git` 自动重建）；v2.3.13 含 torch 运行时不兼容早暴露（launch 预检 torch<2.0 缺 `custom_op` 直接 E701 给清晰原因，install 验收同步 WARN）；v2.3.14 含启动前置硬门禁（`phase8_launch` 进入即综合判定 venv+torch+cuda，`strict=True` 环境不满足直接 E701 且不 spawn 必崩进程，`strict=False` 安装段顺带启动则跳过并诚实汇报，final_summary 据 `launch_skipped_reason` 明确"未启动（环境不满足）"并指引换 GPU 镜像）；v2.3.15 含 **P2 venv 创建健壮性 + 非 CUDA 后端（AMD ROCm）识别**（回应实测 `python3 -m venv` 因 ensurepip 不可用退出码 1 致 `FATAL/unexpected` 崩溃且丢失真实报错：`ensure_venv` 改为标准→`--without-pip`→回退系统 Python 三级尝试并捕获真实 stderr、永不抛错，回退时置 `ctx.venv_fallback` 且 P2 跳过 `requirements.txt` 安装以保护系统环境；`_launch_precheck` 新增 `torch.version.hip` 识别，ROCm 等"非 CUDA 后端"直接诚实跳过启动并提示换 NVIDIA CUDA 镜像）；v2.3.16 含 **P2 依赖安装静默卡死修复**（回应实测"复用 venv 后 `P2/venv 复用已有 venv` 之后长时间无任何日志"：`capture_output=True` 把 pip 输出全吞且超时 1800s，源慢/不可达时像死机；现新增 pip 源可达性预检——pypi.org 不可达则立即报 E203 并指引配置镜像源（如 `pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/`），不再静默卡 30 分钟；并将 pip 安装改为实时流式输出，逐行打印到控制台，用户可见"Downloading …"进度，不再像死机）。v2.3.18 含 **AMD/ROCm（魔搭 AMD 实例）一键支持**：新增 `amd`（BF16 满血，HBM≥80GB）与 `amd-int8`（int8_convrot + 开启 ROCm Triton 后端，HBM 24–80GB）两档，auto 按实测 HBM 自动二选一；`--system-site-packages` 继承魔搭镜像预装 ROCm torch（避免 pip 装回 CUDA 版 → `Found no NVIDIA driver`）、`requirements.txt` 剔除 `torch*` 行、卸载 SageAttention（ROCm 7.2 下 Triton 编译必崩）、启动注入 AMD 优化环境变量（`HSA_OVERRIDE_GFX_VERSION=9.4.2` 等）与 `--highvram`/`--enable-triton-backend` 等参数；AMD 不使用 Veda（Veda 是 NVIDIA nvfp4 专属），改用 ComfyUI 原生 MiniMaxH3 节点（ComfyUI 0.36.0+ 内置，无需额外节点包）。

📚 配套文档：**使用说明** `docs/使用说明.md` ｜ **问题排错** `docs/问题排错.md`

## 1. 快速开始

### 魔搭 Notebook（推荐，首选部署目标）
魔搭持久化目录 `/mnt/workspace`（100GB，关机保留），新开实例无需重装。

**方式 A · 一键 bootstrap（最省事，URL 待定，详见下方"托管地址"）**
```bash
curl -fsSL https://<稳定托管地址>/bootstrap.sh | bash
# 之后每次新开实例，只需：
bash /mnt/workspace/veda-setup/install.sh   # 已安装 → 自动仅启动（MODE_DECISION=launch-only）
```

**方式 B · 手动（git 克隆或 Web IDE 上传 veda-setup/ 目录）**
```bash
cd /mnt/workspace
git clone <脚本仓库地址> veda-setup
bash veda-setup/install.sh
```

安装完成后 ComfyUI 在**后台启动**，终端立即返回并给出访问地址（Web IDE 端口预览 / 隧道，默认 `127.0.0.1:8188`）。选模板 "Veda MiniMax H3 T2VA" 直接出片。

> **许可提示**：MiniMax H3 社区许可**不适用于美国 / 欧盟 / 英国 / 韩国用户**（设计文档 F18），请确认你的使用地域与用途合规；中国个人非商业使用无碍。脚本默认自动接受 License 并记入日志。

### Windows 本地
双击 `install.bat`（或 `python setup_veda.py`），全程无交互，结束后自动尝试打开浏览器。无 Python 时按提示 `winget install Python.Python.3.12` 后重跑。

### macOS（实验性，Apple Silicon）
```bash
bash install.sh   # 走 MLX 后端，模型权重以 ComfyUI 官方支持为准
```

### 魔搭 AMD 实例（ROCm，v2.3.18 新增）
魔搭 AMD 活动实例（如 MI300X/gfx942，约 192GiB HBM）已预装 ROCm 版 PyTorch。`auto` 档会按实测 HBM 自动选择：

- **HBM ≥ 80GB** → `amd`（BF16 满血 DiT + BF16 文本编码器，最佳画质/速度）
- **HBM 24–80GB** → `amd-int8`（int8_convrot 权重 + 开启 ROCm Triton 后端加速）

```bash
cd /mnt/workspace
bash veda-setup/install.sh           # auto：AMD 实例自动选 amd/amd-int8
# 或显式指定：
python veda-setup/setup_veda.py --profile amd        # 大显存 BF16 满血
python veda-setup/setup_veda.py --profile amd-int8   # 小显存 int8 + Triton
```

脚本为 AMD 路径做了关键适配（否则必崩）：

1. **继承系统 ROCm torch**：venv 用 `--system-site-packages`，并剔除 `requirements.txt` 里的 `torch*` 行，避免 pip 装回 CUDA 版（典型报错 `Found no NVIDIA driver`）。
2. **卸载 SageAttention**：ROCm 7.2 下其 Triton kernel 编译失败、采样第一步即崩，ComfyUI 自动回退 PyTorch attention。
3. **AMD 优化环境变量**：`HSA_OVERRIDE_GFX_VERSION=9.4.2`、`PYTORCH_TUNABLEOP_ENABLED=1`（编译缓存持久化到云盘 `run/../.cache/torch_tunableop`）、`MIOPEN_FIND_MODE=FAST` 等。
4. **启动参数**：`amd` 仅 `--highvram`；`amd-int8` 加 `--enable-triton-backend --disable-smart-memory --disable-xformers --use-pytorch-cross-attention`。
5. **不用 Veda**：Veda 是 NVIDIA nvfp4 专属；AMD 走 ComfyUI 原生 MiniMaxH3 节点（`MiniMaxH3ImageToVideo` 等，ComfyUI 0.36.0+ 内置），在 UI 模板库直接加载 MiniMax-H3 工作流（T2V / ReferenceToVideo）即可。

> 参考实践：魔搭社区两篇部署手记（BF16 满血 / int8+Triton 低显存）。详见脚本内 `lib/platform.py`、`lib/phases.py` 的 `rocm` 分支与 `assets_manifest.json` 的 `h3-*-bf16` / `h3-*-int8` 资产条目。

## 2. 常用参数

| 参数 | 说明 | 默认 |
|---|---|---|
| `--install-root PATH` | 安装根目录 | 自动探测（魔搭 → `/mnt/workspace/comfyui-veda`） |
| `--mode NAME` | 运行模式：`auto`=自动判定（已安装仅启动 / 未安装安装+启动）；`install`=强制安装段（幂等增量）；`launch`=强制仅启动 | `auto` |
| `--only pN[,pN]` | 单环节执行：`--only p5` 补模型、`--only p1,p3` 重装 ComfyUI+节点；P0 前置、不改 `install_completed`、未含 p8 不启动 | 无 |
| `--profile NAME` | `minimal`/`standard`/`full`/`lowvram`/`auto`；AMD/ROCm 实例另加 `amd`(BF16 满血,≥80GB) / `amd-int8`(int8+Triton,24–80GB) | `auto`（按实测 VRAM/HBM 选） |
| `--skip-models` | 只搭环境不下模型 | false |
| `--keep-cache` | 下载后保留模型缓存（默认清理省磁盘） | false |
| `--listen` | 绑定 `0.0.0.0`（默认 `127.0.0.1`） | false |
| `--smoke-test` | 无头启动探测 `/object_info` | false |
| `--no-launch` | 装完不启动 ComfyUI | false |
| `--doctor` | 只生成诊断包后退出 | false |
| `--dry-run` | 只打印动作不落盘 | false |

完整参数见 `setup_veda.py` 的 `--help`，设计原则见设计文档 §7.2。

## 3. 故障排查

任何失败都会**退出非 0 + 打印错误码 + 诊断包路径**。先执行：
```bash
python setup_veda.py --doctor
```
把生成的 `logs/diagnose_*.tar.gz`（Windows 为 `.zip`）附上提 issue 即可，无需手动收集信息。

常见错误码：`E102` GPU 算力不足；`E104` 磁盘不足（会自动降档）；`E201` ComfyUI 下载失败（GitHub 不可达，检查 gh-proxy）；`E401` 模型资产下载失败（重跑续传或 `--only p5` 单补模型；历史落点 bug 已在 v2.3.7 修复并自动回收错位文件）；`E603` 并发锁冲突（删除残留 `.setup.lock` 或等待另一安装）；`E701` 显式 `--mode launch` 但安装未完成（去掉该参数即可，auto 会自动安装/修复）。失败重跑很快：已对齐锁定 commit 的环节（P1/P3）零网络操作直接跳过，仅补真正缺失项。

> 完整错误码表、按阶段排错、网络/磁盘/GPU/锁/校验等专项排查，见 **`docs/问题排错.md`**。

## 4. 实现状态 / 待实测

代码已实现 M1–M6（驱动 + P0–P8 全流程 + 安装/启动分离）。以下为**前置验证清单（设计文档 §16，G1–G7）**：**G1/G2/G3 已在本地实测确认**，G4–G7 仍需真实魔搭实例验证（本仓库已把相关行为做成运行时自适应探测）：

- **G1** ✅ 已确认：`Comfy-Org/MiniMax-H3` 在 ModelScope 存在（Downloads≈386万，目录结构与 HF 一致，主模型可走魔搭源）
- **G2** ✅ 已确认：Veda 预测器两镜像在 ModelScope 存在（`Veda-Sparse/Minimax-H3-T2VA-Veda-8NFE-600Step-Preview`、`Veda-Sparse/Minimax-H3-R2VA-Veda-Preview`，各 ~262MB，P4 主源走魔搭）
- **G3** ✅ 已确认：ComfyUI（pin `7f7fd918…`，2026-10）内建 MiniMax-H3 原生加载器；第三方 `ComfyUI-MiniMaxH3` 节点经核实**不存在（GitHub 404）**，已废弃方案 B，统一走原生（§8 P3）
- **G4** `modelscope download --local_dir` 在 `/mnt/workspace` 是否复制缓存（影响 E1 缓存预算，§6 规则 8）
- **G5** 魔搭内 `gh-proxy.com`/`gitclone.com` 是否可达（影响 GitHub 兜底）
- **G6** A10 24GB 实测能否跑 `standard` 档（决定默认档位）
- **G7** 魔搭本地盘（`/root`）容量是否 ≥ 最大模型体积（决定缓存重定向策略，§6 规则 8b）

> manifest 中 ComfyUI 与 Veda 的 `commit` 已回填 G1/G2 实测的真实值（`7f7fd918…` / `67e4eb55…`，均为 2026-10），锁定可复现；G1/G2/G3 已确认，G4–G7 仍需魔搭实例实测。

## 5. 托管地址（开放项）

`install.sh` / `bootstrap.sh` 需要一个可访问的托管地址（GitHub / GitLab / 魔搭 / OSS）。当前在本地开发，可先以"Web IDE 上传 `veda-setup/` 目录"或"bootstrap 托管"方式交付。
