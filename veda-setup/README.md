# Veda-on-ComfyUI 一键搭建脚本

在魔搭社区（ModelScope）免费 Notebook 等环境**一键部署 Veda-on-ComfyUI**（MiniMax-H3 视频扩散模型的官方加速节点，学习型稀疏注意力，约 3× 提速）。

> 设计文档：`docs/Veda一键脚本设计文档.md`（v2.3.6）。本脚本严格按设计文档实现：多平台、持久化免重装、**安装/启动自动分流**、全自动、优先魔搭下载能力、结构化日志 + 诊断包。

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

## 2. 常用参数

| 参数 | 说明 | 默认 |
|---|---|---|
| `--install-root PATH` | 安装根目录 | 自动探测（魔搭 → `/mnt/workspace/comfyui-veda`） |
| `--mode NAME` | 运行模式：`auto`=自动判定（已安装仅启动 / 未安装安装+启动）；`install`=强制安装段（幂等增量）；`launch`=强制仅启动 | `auto` |
| `--profile NAME` | `minimal`/`standard`/`full`/`lowvram`/`auto` | `auto`（按实测 VRAM 选） |
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

常见错误码：`E102` GPU 算力不足；`E104` 磁盘不足（会自动降档）；`E201` ComfyUI 下载失败（GitHub 不可达，检查 gh-proxy）；`E401` 模型资产下载失败；`E603` 并发锁冲突（删除残留 `.setup.lock` 或等待另一安装）；`E701` 显式 `--mode launch` 但安装未完成（去掉该参数即可，auto 会自动安装/修复）。

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
