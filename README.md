# MiniMax-H3 One-Click Setup for ModelScope Notebooks

> 中文简介见下方「[项目简介](#项目简介)」。

A zero-to-run installer that deploys **ComfyUI + MiniMax-H3** on a
[ModelScope (魔搭) Notebook](https://modelscope.cn) with a single command. It
auto-detects the GPU vendor and picks the right backend:

| GPU | Backend | Stack |
|-----|---------|-------|
| **NVIDIA** (CUDA) | Veda sparse-acceleration | `Veda-on-ComfyUI` nodes + `nvfp4` / `int8_convrot` weights |
| **AMD** (ROCm)    | Native MiniMaxH3 nodes | ComfyUI 0.36.0+ built-in `MiniMaxH3ImageToVideo` + `BF16` / `int8_convrot` weights |

The script is idempotent, verifies every model with a safetensors header check,
never re-downloads a complete file, and refuses to run on hardware that cannot
physically fit the model (see [Known Limitations](#已知限制)).

---

## 项目简介

本项目提供一个**一键脚本**，在魔搭（ModelScope）Notebook 上自动部署
**ComfyUI + MiniMax-H3**（MiniMax 最新的图像/视频生成模型），让你无需手动拼装
环境、装节点、下权重，即可在浏览器里用 ComfyUI 跑文生视频 / 参考图生视频。

核心特性：

- **跨显卡**：自动识别 NVIDIA / AMD(ROCm)，分别走 Veda 加速 / ComfyUI 原生节点。
- **自动选档**：按实测显存与内存选择 `standard` / `full` / `lowvram` / `amd` / `amd-int8`。
- **幂等 & 完整校验**：已存在且头部完整的模型直接跳过，不重复下载；缺模型自动补齐。
- **容量门禁**：显存/内存不足时（如 24GB 的 A10）直接拒绝并提示换卡，而不是跑到推理才崩。
- **原生 MiniMax-H3 支持**：AMD 实例无需 Veda（Veda 是 NVIDIA nvfp4 专属），走 ComfyUI 内置节点。

---

## 目录结构

```
.
├── README.md                 # 本文件（项目总览）
├── .gitignore
├── veda-setup/               # ★ 一键安装脚本（核心）
│   ├── install.sh            # Linux/macOS 启动入口
│   ├── install.bat           # Windows 启动入口
│   ├── setup_veda.py         # 主程序（P0–P8 流水线）
│   ├── assets_manifest.json  # 模型资产清单（文件名 + min_bytes + 下载源）
│   ├── profiles.json         # 档位定义（standard/full/lowvram/amd/amd-int8…）
│   ├── test_v2310.py         # 自检集成测试（72 项，全 mock，无外部依赖）
│   ├── lib/
│   │   ├── phases.py         # P0 预检→P8 启动 全流程编排 + 各后端分支
│   │   ├── platform.py       # 平台 / GPU / venv / 内存 探测
│   │   ├── assets.py         # 模型下载 + safetensors 头部完整性校验
│   │   └── diag.py           # 状态机 + 诊断包
│   └── README.md             # 脚本详细文档（变更记录 / 参数 / 排错）
└── docs/                     # 中文设计文档与使用说明
    ├── Veda一键脚本设计文档.md
    ├── 使用说明.md
    ├── 问题排错.md
    └── 设计方案评审与优化建议.md
```

---

## 快速开始

在**魔搭 Notebook**（已预装 ROCm/CUDA 版 PyTorch 的实例）的终端里：

```bash
git clone <this-repo> && cd <this-repo>
bash veda-setup/install.sh
```

脚本会：

1. 探测平台（是否魔搭环境）、Python、GPU 厂商与显存；
2. 选档（`auto` 默认按实测显存/内存自动选，也可手动 `--profile`）；
3. 拉取 ComfyUI 与所需节点、建 venv、装依赖；
4. 下载并校验模型权重（仅缺失/不完整才下载）；
5. 后台启动 ComfyUI，打印访问地址。

在魔搭里通过「Web IDE 端口预览 / 隧道」打开 `http://127.0.0.1:8188` 即可使用。

### 手动指定档位

```bash
# NVIDIA 满血（需 A100 80GB 级）
python veda-setup/setup_veda.py --profile full

# AMD 大显存（≥80GB HBM，BF16 满血）
python veda-setup/setup_veda.py --profile amd

# AMD 中显存（24–80GB HBM，int8_convrot + Triton）
python veda-setup/setup_veda.py --profile amd-int8

# 仅补齐模型（已装好环境、只想补权重）
python veda-setup/setup_veda.py --only p5

# 干跑（只打印动作、不落盘）
python veda-setup/setup_veda.py --dry-run
```

---

## 档位（Profiles）

| 档位 | 后端 | 显存门槛 | 内存门槛 | 磁盘 | 说明 |
|------|------|---------|---------|------|------|
| `minimal` | NVIDIA | ≥12GB | — | ~0.3GB | 仅 Veda 节点 + 预测器，不含主模型 |
| `standard` | NVIDIA | ≥24GB | ≥48GB | ~59.5GB | int8 DiT(FL2VA+R2VA) + 32B nvfp4 编码器 + R2VA 全套 |
| `full` | NVIDIA | >24GB | ≥48GB | ~59.5GB | 同 `standard` 模型集，画质优先 |
| `lowvram` | NVIDIA | 8–16GB | — | ~36.4GB | 第三方量化权重，需 `--accept-third-party` |
| `amd` | AMD/ROCm | ≥80GB HBM | — | ~127GB | BF16 满血 DiT + bf16 文本编码器 |
| `amd-int8` | AMD/ROCm | 24–80GB HBM | — | ~66GB | int8_convrot DiT + int8 文本编码器 + Triton 后端 |

> **注意**：所有 NVIDIA 带模型档（standard/full/lowvram）共用同一个
> ~20GB 的 `int8_convrot` DiT，因此 **24GB 显存卡（如 A10/RTX3090）在当前
> 权重集下无法稳定跑通**（推理首步即被 OOM-killer 静默杀掉）。脚本内置
> **容量门禁 E104** 会在安装/启动阶段直接拦截并提示换 A100 80GB，而非下载
> 数十 GB 后才崩溃。

---

## 命令行参数（节选）

| 参数 | 作用 |
|------|------|
| `--profile NAME` | `minimal`/`standard`/`full`/`lowvram`/`amd`/`amd-int8`/`auto` |
| `--mode auto\|install\|launch` | 自动 / 仅安装 / 仅启动已装实例 |
| `--only pN[,pN]` | 仅执行指定环节（如 `--only p5` 补模型） |
| `--skip-models` | 跳过主模型资产下载 |
| `--accept-third-party` | 允许 `lowvram` 档的第三方量化权重 |
| `--listen` | 绑定 `0.0.0.0`（默认 `127.0.0.1`） |
| `--dry-run` | 只打印动作、不落盘 |
| `--smoke-test` | 无头启动探测 `/object_info` |
| `--doctor` | 仅生成诊断包后退出 |
| `--resume` / `--force` / `-y` | 续跑 / 覆盖外来 ComfyUI 检测 / 全部默认 yes |
| `--source-priority auto\|modelscope\|hf` | 下载源优先级（魔搭主源 + hf-mirror 兜底） |

完整说明见 [`veda-setup/README.md`](veda-setup/README.md)。

---

## 自检与测试

```bash
cd veda-setup
python test_v2310.py        # 72 项集成测试（全 mock，无外部依赖）
```

脚本自身也在每次运行做内置自检：JSON 合法性、档位→资产引用完整性、
`min_bytes` 下界、模型落点冲突、资产 URL 可达性（HEAD）、`py_compile` 语法，
以及 AMD/NVIDIA 后端分支的单元级回归。

---

## 已知限制

- **A10 / 24GB NVIDIA 显卡无法运行**（容量不足，见上表）。需 A100 80GB 级实例。
- AMD 路径要求 **ComfyUI ≥ 0.36.0**（原生 MiniMaxH3 节点）；脚本会在验收阶段检查。
- AMD 路径**不使用 Veda**（Veda 为 NVIDIA nvfp4 专属），改用 ComfyUI 原生节点。
- 模型权重体积较大（BF16 满血档约 127GB 磁盘），请确认实例磁盘与下载带宽。
- 设计上 `h3-te-nvfp4` 与 `h3-te-awq` 两个资产指向同一物理文件
  （`qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`），二者分属互斥档位
  `standard` / `lowvram`，属有意为之，非冲突。

---

## 许可证

本项目以 **MIT 许可证** 开源。如需调整为其他许可证，请修改根目录 `LICENSE`
文件（此处未附，可依据实际需求添加）。

---

## 参考与致谢

- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- [Veda-on-ComfyUI](https://github.com/veda-sparse/Veda-on-ComfyUI)（NVIDIA 加速节点）
- [Comfy-Org/MiniMax-H3](https://modelscope.cn/models/Comfy-Org/MiniMax-H3)（官方权重 / 原生加载器）
- 魔搭社区两篇部署手记（AMD/ROCm 路径设计参考）
- [ModelScope 魔搭](https://modelscope.cn) 提供的预装 PyTorch 实例
