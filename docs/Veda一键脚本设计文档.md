# Veda-on-ComfyUI 一键搭建脚本 · 设计文档

| 项目 | 内容 |
|---|---|
| 文档版本 | v2.3.14（v2.3.13 + **启动前置硬门禁（回应实测"安装成功却启动必崩"循环）**：① 新增 `_launch_precheck(ctx)`——综合 venv 存在性 + `_torch_compat_ok`（torch>=2.0 / `custom_op`）+ `torch.cuda.is_available()`，一次性判定环境是否满足 ComfyUI 运行条件；② `phase8_launch(ctx, strict)` 加启动前置门禁——`strict=True`（`--mode launch` / `--only p8`）环境不满足直接抛 **E701** 并给出清晰原因（绝不 spawn 必崩进程、等 120s、再 dump traceback）；`strict=False`（auto/install 安装段顺带启动）环境不满足则**跳过启动**并诚实汇报（`ctx.launch_skipped_reason`），安装本身已完成、不崩溃、不谎报；③ 驱动层——`--mode install`/auto 降级安装路径改用 `phase8_launch(ctx, strict=False)`，`--mode launch`/`--only p8` 保持 `strict=True`；④ `final_summary` 在 `launch_skipped_reason` 已设置时明确"未启动（环境不满足）"并指引换 GPU 镜像后 `run/start.sh` 启动） |
| 日期 | 2026-10-12 |
| 状态 | 设计定稿（假设已清零：G1–G3 / F11 / F17 / F18 / F19 已实测确认；G4–G7 为运行时探测项，非设计层假设）；v2.3.5 安装/启动分离已随 v2.3.6 完成代码实现（M6 ✅）；v2.3.7 为首次魔搭实例实测反馈修复（E401 落点/源链 + 阶段级快速跳过 + `--only`），代码随本版同步实现；v2.3.8 为全量代码自检修正（P8 守护/会话脱离/tarball 幂等/通用链兜底等 10 项），代码随本版同步实现；v2.3.9 为无 GPU 交互继续修复（GPU 预检不再直接退出，改询问/`--yes` 继续）；v2.3.10 为同类"可询问跳过"环节扩展（E102 算力不足/E601 外来 ComfyUI 改交互询问）与自检修正（compute_cap_ok 死代码、_file_tree Windows 分隔符、P7 无 CUDA 误报、P0 死返回值清理）；v2.3.11 为 P8 启动诚实化修复（启动崩溃/超时不再谎报成功：崩溃抛 E701 + 打印 `run/comfyui.log` 尾部，超时诚实标记未就绪；launch 预检补 CUDA 可用性早期信号）；v2.3.12 为同类"谎报/不完整"缺陷自检修复（final_summary 三态——未启动不再谎报可访问 URL；safetensors 数据段完整性校验防"下载不完整却判定成功"；download_model_asset 回收完整文件 + 清理半截 target；损坏 .git 自动重建）；v2.3.13 为 torch 运行时不兼容早暴露（新增 `_torch_compat_ok` 探针，launch 预检 torch<2.0 缺 `custom_op` 时直接 E701 给出清晰原因而非崩溃后 dump traceback；install 验收同步 WARN）；v2.3.14 为启动前置硬门禁（新增 `_launch_precheck` 综合判定 venv+torch+cuda，phase8_launch 加 `strict` 门禁——strict=True 环境不满足直接 E701 不 spawn 必崩进程；strict=False（安装段顺带启动）环境不满足则跳过启动并诚实汇报 launch_skipped_reason，安装已完成不崩溃不谎报；final_summary 据 launch_skipped_reason 明确"未启动（环境不满足）"并指引换 GPU 镜像） |
| 关联文档 | 评审与优化建议见 `docs/设计方案评审与优化建议.md` |
| 首选部署目标 | **魔搭社区（ModelScope）Notebook 免费实例**，持久化目录 `/mnt/workspace` |
| 本地开发工作区 | `D:/ai_desktop_project/buddy/modelscope_comfyui` |

---

## 1. 背景与目标

### 1.1 背景

参考文章《砍掉90%计算，让MiniMax-H3速度暴涨3倍！这个开源神仙插件》（2026-10-11）介绍了 **Veda-on-ComfyUI**：MiniMax-H3 视频扩散模型的官方 ComfyUI 自定义节点，通过学习型稀疏注意力（ICML 2026）跳过约 90% 无效注意力计算，12GB 显卡端到端提速约 2.9 倍。

手工流程 5 步（环境检查 → 装节点 → 下预测器 → 组装工作流 → 调参避坑）对新手不友好，且模型资产合计约 42GB（standard 档，已按魔搭官方清单校准），国内直连 HuggingFace 困难。

### 1.2 目标（v2.0 重新定义，v2.1/v2.2 自检修订）

> **人工只做一件事：运行一条命令。** 其余全部自动完成，并可重复执行、跨实例复用。该命令自动判定安装状态：**持久盘上尚无安装（首次使用）→ 自动安装并启动**；**已有安装（含关机后新开的实例，因其挂载同一持久盘）→ 自动仅启动**（跳过安装段，秒级进入 ComfyUI），无需人工区分"安装"还是"启动"。

1. **主流平台全覆盖**：魔搭 Notebook（Linux，首选）/ Windows / 通用 Linux / macOS（实验性）；
2. **默认部署到魔搭持久化目录** `/mnt/workspace`：实例关机后数据不丢，**新开实例无需重装**（幂等检测，分钟级恢复）；
3. **全自动化、零人工参与**：所有确认项均有默认值，无人值守模式为默认行为；
4. **优先复用魔搭社区能力**：模型下载走 ModelScope 官方 CLI/SDK（国内直连、免翻墙），镜像预装的 Python/PyTorch 直接复用，不重复下载；
5. **完善日志与诊断**：结构化日志 + 错误码 + 一键诊断包，出问题可快速定位。

### 1.3 非目标

- 不替代 ComfyUI 日常使用，不常驻后台管理；
- 不修改/微调任何模型权重（Veda 本身即非侵入式）；
- 不做模型量化转换。

---

## 2. 参考资料与事实基线

| # | 事实 | 来源 |
|---|---|---|
| F1 | ComfyUI 版本必须 **≥ 0.38.0** | 文章 + GitHub README |
| F2 | 显卡要求 NVIDIA **SM80+**；Apple M 系列走 MLX 后端 | 文章 + README |
| F3 | 节点仓库 `https://github.com/veda-sparse/Veda-on-ComfyUI.git`；安装方式：Manager 搜 "Veda" / `comfy node install veda-sparse-attention` / git clone 到 `custom_nodes/` | 文章 + README |
| F4 | 预测器（275MB FP8）手动放 `ComfyUI/models/veda/`：T2VA/FL2VA 用 `minimax_h3_t2va_veda_8nfe_600step_preview_fp8.safetensors`；R2VA 用 `minimax_h3_r2va_veda_preview_fp8.safetensors`；节点自身不联网 | 文章 + README |
| F5 | 预测器 HF 仓库 `Veda-Sparse/Minimax-H3-T2VA-Veda-8NFE-600Step-Preview` | 文章 |
| F6 | 节点形态 MODEL in → MODEL out，接在 LoRA 之后、guider/sampler 之前 | README |
| F7 | 避坑：不可与 ComfyUI 原生 Model Sparse Attention 节点同链混串；Mem Eff Sage Attention Patch 类全局替换节点须在 Veda **之前** | README |
| F8 | 避坑：Windows "no sparse kernel works" → 补装 `triton-windows`；状态栏显示 `Triton INT8 (SMxxx)` 才算生效 | 文章 + README |
| F9 | MiniMax-H3 官方 ComfyUI 权重：HF `Comfy-Org/MiniMax-H3`，**魔搭镜像 `Comfy-Org/MiniMax-H3` 已确认存在（G1 实测：`Downloads≈386万`、`License=minimax-h3-community-license-agreement`、`StorageSize≈525GB` 含多档量化；目录含 `diffusion_models/ text_encoders/ vae/ loras/ embeddings/ model_patches/`，文件名与 HF 一致）** | 魔搭社区模型页（G1 已复核） |
| F10 | 推荐低显存组合（12–16GB）：`minimax_h3_fl2va_pruned_int8_convrot` + `qwen3vl_32b_minimax_h3_nvfp4_awq` + 双 VAE + 8 步 Turbo LoRA | HF/魔搭 README |
| F11 | **魔搭 Notebook 免费实例**：100GB 持久化存储挂载 `/mnt/workspace`（关机保留）；A10 GPU 24GB；Web IDE；镜像预装 Python + PyTorch + ModelScope；`/tmp`、`/root`、`/home` 等非 `/mnt` 路径停机即清空；默认 root | 魔搭官方文档 `modelscope.cn/docs/notebook/intro`（2026-10-12 复核） |
| F12 | ModelScope 下载能力：CLI `modelscope download --model <repo> --include <glob> --local_dir <dir>`；SDK `snapshot_download(model_id, allow_patterns=, local_dir=)` / `model_file_download`；缓存目录可用 `MODELSCOPE_CACHE` 改 | 魔搭官方文档 |
| F13 | License：`minimax-h3-community-license` | HF/魔搭页面 |
| F14 | **standard 档 VRAM 占用为运行时实测项（非设计假设）**：standard 档（int8 DiT + 32B nvfp4 编码器）在魔搭 A10 24GB 的 VRAM 可行性待首实例实测（G6）；设计层**不预设结论**——P0 按实测显存映射档位（<16GB→lowvram / 16–24GB→standard / >24GB→full），不足时自动降档兜底。文章实测卡为 RTX 5070 12GB；int8 档位"12–16GB 可跑"来自第三方指南 | G6 / 自检识别 |
| F15 | 工程经验：国内云环境访问 github.com 不稳定（超时/重置），ComfyUI 与 Veda 节点均托管于 GitHub，克隆需有非 git 兜底 | 工程实践 |
| F16 | modelscope CLI 与 huggingface-cli 均支持断点续传；免费 GPU 额度可能中途耗尽，下载必须可中断可续 | 魔搭文档 + 工程实践 |
| F17 | **魔搭 Notebook 外网访问受限，官方文档明确"外网访问受限，包括 huggingface 等"**（原文 `modelscope.cn/docs/notebook/intro`：网络访问 — 外网访问受限，包括 huggingface 等）；故魔搭内 HF/HF-mirror 不可依赖，模型第一源必须走魔搭 ModelScope | 魔搭官方文档 `modelscope.cn/docs/notebook/intro`（2026-10-12 复核） |
| F18 | **MiniMax H3 社区许可（minimax-h3-community-license）将美国/欧盟/英国/韩国排除在"适用地域（Applicable Territory）"之外**，许可授予明确限定于适用地域；非商业免费，商业产品/服务年收入 >$20M 需书面授权，商业界面须标注 "MiniMax H3"；Qwen3-VL-32B 文本编码器单独以 Apache-2.0 提供 | MiniMax H3 LICENSE 原文（HuggingFace `MiniMaxAI/MiniMax-H3/LICENSE`，Excluded Territories = EU/UK/ROK/USA；多源复核 2026-10-12） |
| F19 | 国内可用 GitHub 代理（第三方、可能随时失效，需多候选+可配置）：经 2026-10-12 实测可达 `gh-proxy.com`（200）/ `ghproxy.net`（200）/ `gitclone.com`（200）/ `hf-mirror.com`（200）；**`mirror.ghproxy.com` 实测不可达（HTTP 000，已废弃，不纳入候选链）** | 实测 curl 探测 2026-10-12 |
| F20 | **lowvram 档的"社区第三方量化权重"源尚未在魔搭/HF 确认存在**：manifest 当前未内置任何第三方量化源；`--accept-third-party` 为预留开关，启用前须先确认可用源，设计不假设其可下载（不依赖该档即可达标；lowvram 当前实际复用官方 nvfp4_awq TE 14.6GB） | 自检识别（避免假设） |

---

## 3. 部署目标矩阵（平台分级）

| 层级 | 平台 | 定位 | 默认安装根 | 特殊处理 |
|---|---|---|---|---|
| **T1 首选** | 魔搭 Notebook（Linux，A10 24GB，`/mnt/workspace` 持久化） | 零成本体验/生产主力 | `/mnt/workspace/comfyui-veda` | 复用镜像预装 torch/modelscope；venv 用 `--system-site-packages`；pip 缓存落持久盘 |
| T2 | Windows 10/11（独显 SM80+） | 本地创作 | `<脚本目录>/ComfyUI`（可 `--install-root` 改） | Triton 缺失自动装 `triton-windows`；`.bat` 双击入口；无 Python/git 时有明确指引 |
| T3 | 通用 Linux 服务器（NVIDIA SM80+） | 自有机房 | `~/comfyui-veda` | 同 T1 逻辑 |
| T4 实验性 | macOS Apple Silicon（MLX 后端） | 尝鲜 | `~/comfyui-veda` | 走 MLX；模型权重以 ComfyUI 官方支持为准；文档明确标注实验性 |

**安装根自动探测顺序**（无需人工指定）：
1. 命令行 `--install-root` 显式指定 → 最高优先；
2. 环境变量 `COMFYUI_INSTALL_ROOT`；
3. 检测到 `/mnt/workspace`（魔搭实例）→ `/mnt/workspace/comfyui-veda`；
4. Windows → `<脚本所在目录>/ComfyUI`；其他 → `$HOME/comfyui-veda`。

---

## 4. 需求 → 设计决策映射

| 新需求 | 设计决策 |
|---|---|
| 主流平台 | **单一 Python 驱动**（`setup_veda.py`，仅用标准库）+ 平台薄入口（`install.bat` / `install.sh`）。Python 是 ComfyUI 的硬依赖、必然存在，一套逻辑跑三平台，避免 PowerShell/Bash 双实现漂移 |
| 魔搭持久化目录 | 根目录自动探测 `/mnt/workspace`；**ComfyUI、venv、模型、节点、状态文件、日志全部落安装根内**；pip 缓存与 ModelScope 缓存也重定向到持久盘 |
| 关机不丢/新实例免重装 | 状态文件 `<root>/.setup-state.json` **以单资产为最小粒度**记录完成态与指纹；重跑即幂等恢复：已完成的全跳过，只补缺失部分；venv 失效（如镜像 Python 版本变化）才重建；**魔搭环境下 venv 与代码同样落 `/mnt/workspace`，`--local-code` 强制关，避免本地盘关机清空导致每次重建（E3，见 §6 规则7）** |
| 全自动零人工 | 默认非交互；License 自动接受（记入日志，`--interactive` 可退回人工确认）；体积确认、目录采用、覆盖备份全部有默认行为；失败时给出唯一错误码 + 诊断包路径 |
| 优先魔搭能力 | 模型下载第一源为 ModelScope CLI（`pip install modelscope` 按需）；镜像预装 torch 直接复用；**源链自适应**（P0 探测 egress）：魔搭实例内 huggingface.co 直连受限（F17，2026-10-12 实测）但 hf-mirror 可达——模型源链 `modelscope → hf-mirror → gh-proxy 代理 HF(末位)`、代码源链 `github 直连(首位) → gh-proxy → codeload(末位)`；通用环境为 `modelscope → hf-mirror → huggingface`（E5/E6，详见 §9） |
| 日志与定位 | JSONL 结构化日志 + 控制台镜像；错误码注册表；`--doctor` 一键生成诊断包（环境快照+日志+资产指纹+网络探测） |
| 健壮性（v2.3 增强） | **降级链优先于失败**：每个外部依赖都有 Plan B（见 §5、§10）；并发锁含 **stale PID 检测自动释放**（避免崩溃后死锁）；per-asset 断点续传；GitHub 类源统一走**多候选可达代理**（gh-proxy.com 等，E6） |
| 安装/启动分离 + 已安装自动判定（v2.3.5） | 单一命令 `auto` 模式：据 `.setup-state.json` 的 `install_completed` 标志 + 目录存在性自动判定「已安装」→ 仅启动（跳过 P1–P7，不重下/不重装）；「未安装」→ 安装段（P0–P7）后启动。另提供显式 `--mode install`（强制进入安装段，**幂等增量**——仅补缺失/过期项，不重下已完成资产）/`--mode launch`（强制仅启动，未完成报 E701）。`run/start.sh` 作为纯启动入口（已装实例可独立调用） |

---

## 5. 总体架构

```
┌─────────────────── 人工唯一入口（一条命令）───────────────────┐
│  魔搭 Notebook:  bash install.sh                             │
│  Windows:        双击 install.bat                            │
│  任意平台:        python setup_veda.py                        │
└───────────────────────────┬──────────────────────────────────┘
                            ▼
              setup_veda.py（唯一驱动，Python 标准库）
                            │
                   ┌────────┴─────────┐
                   │  M0 运行模式判定   │  ← 新增（v2.3.5）：读 .setup-state.json
                   │  auto/install/launch│     install_completed + 目录存在性
                   └────────┬─────────┘
              ┌─────────────┴─────────────┐
       已安装（auto/launch）          未安装（auto/install）
              │                              │
              ▼                              ▼
       轻量预检 → P8 启动            P0 预检 → P1–P7 安装段 → P8 启动
     （跳过 P1–P7，秒级进入）         （写 install_completed=true 后启动）
                            │
      ┌─────────────┬───────┴───────┬──────────────┐
      ▼             ▼               ▼              ▼
 assets_manifest.json  profiles.json   .setup-state.json  logs/*.jsonl
 （声明式资产清单      （Profile→        （幂等状态，        （结构化日志+
  + 多源候选）          资产集合）         跨实例恢复）         诊断包）
```

**设计要点**：
- **驱动与资产分离**：所有"下载什么、放哪、从哪下"都在 `assets_manifest.json` 中声明，脚本只负责编排。上游模型更版只改 JSON，不动代码；
- **薄入口**：`install.bat` / `install.sh` 只做"找到 Python → 调 `setup_veda.py`"，自身零业务逻辑（见 §7.5 跨平台约束）；
- **零第三方依赖**：驱动本身只用标准库（subprocess/json/logging/hashlib），在全新实例上开箱即用；
- **降级链优先于失败**：git clone → tarball 镜像解压；venv 复用 → 重建；modelscope → hf-mirror → HF 官方。单点故障不阻断整体（详见 §8、§10）。

---

## 6. 持久化与目录设计（核心）

```
<安装根>/                                  # 魔搭实例 = /mnt/workspace/comfyui-veda
├── ComfyUI/                              # 本体（git 仓库或 tarball 解压，可升级）
│   ├── models/
│   │   ├── veda/                         # 预测器 ×2（~525MB：每个 ≈262MB，魔搭实测）
│   │   ├── diffusion_models/  text_encoders/  vae/  loras/
│   └── custom_nodes/
│       ├── Veda-on-ComfyUI/              # 加速节点（MODEL in→out）
│       └── （MiniMax-H3 加载器走 ComfyUI 原生，无需单独 clone 节点；`Comfy-Org/MiniMax-H3` 官方带 `comfyui` / `diffusion-single-file` 标签）
├── venv/                                 # 虚拟环境（持久化，pip 包不丢）
├── .cache/
│   ├── pip/                              # PIP_CACHE_DIR（加速 venv 重建）
│   └── modelscope/                       # MODELSCOPE_CACHE（下载缓存/续传）
├── .setup-state.json                     # 幂等状态（跨实例恢复的依据）
├── .setup.lock                           # 并发锁（防两次安装互相踩踏）
├── run/                                  # 运行期：start.sh stop.sh status.sh + comfyui.log
└── logs/                                 # setup_*.jsonl + diagnose_*.tar.gz
```

**持久化规则**：
1. 一切"重启后还要用"的东西必须在安装根内（venv、模型、缓存、脚本自身、日志）；
2. 脚本交付物本身也放 `/mnt/workspace`（如 `/mnt/workspace/veda-setup/`），新实例免重新上传；
3. **二次安装（新实例）耗时目标 ≤ 5 分钟**：跳过已完成项，仅做预检+验收；venv 复用；模型文件做"存在性+大小+快速指纹"校验，不重算全量哈希（`--deep-verify` 才全量）；**安装段（P0–P7）全部成功结束后，在 state 顶层写入 `install_completed=true` 与 `install_profile`，作为「已安装」判定的唯一权威依据（§8 M0）**；**进入安装段时先将 `install_completed` 复位为 `false` 并落盘**（防止 manifest 更新后的重装半途失败时，旧标志残留导致 M0 误判「已安装」、直接启动缺资产的实例），全部成功后再置 `true`；
4. 镜像 Python 版本变化导致 venv 失效时：仅重建 venv（pip 缓存命中，分钟级），不动模型；
5. **state 粒度到单资产**：每个资产独立记录 `id / status / 大小 / 快速指纹 / 所用源`，P5 下约 42GB（standard 档）分文件中断后精确续传；state 同时记录 `platform`、`profile`、`manifest_sha1`，任一变化即重新评估相关资产，避免上游更名误判"已安装"；state 损坏时备份为 `*.corrupt-<ts>` 并按新装处理；
6. **并发保护**：安装根内置 `.setup.lock`（Linux/macOS 用 flock，Windows 用 msvcrt/O_EXCL lockfile），并行二次运行直接退出并提示（错误码 E603）；**锁含 PID/心跳，stale 检测**：持有者进程已死则自动释放，避免崩溃后永久死锁（E603 误报）；
7. **`--local-code` 在魔搭环境强制默认 OFF（E3）**：魔搭实例 `/root`、`/tmp`、本地盘等非 `/mnt` 路径**关机即清空**，把 venv/代码放本地盘 = 新实例必重建 venv，**直接违背"免重装"**。故魔搭环境（检测到 `/mnt/workspace`）下 `--local-code` 强制关闭，全部内容（含 venv）落 `/mnt/workspace`；该选项仅作为 T2/T3（本地盘持久）的可选逃生舱；
8. **缓存可能翻倍，须计入磁盘预算（E1，关键）**：`modelscope download --local_dir` 在 NAS（`/mnt/workspace`）上常因跨文件系统回退为 **copy**，导致模型在缓存与目标各一份（以 standard 档 42GB 模型为例：缓存 42GB+目标 42GB=84GB，加 ComfyUI/venv 约 88GB 已逼近 100GB 上限；full 档或多模型必超，且 NAS 写放大会进一步吃空间）。**注意**：仅"校验后清理缓存"救不了峰值——爆盘发生在下载完成前（缓存+目标累计超 100GB 时即写失败），故必须双管齐下：**（a）下载校验通过后默认清理冗余 `.cache/modelscope`**（`--keep-cache` 可关）；（**b）更关键——当安装根在 NAS 且容量紧张时，自动将 `MODELSCOPE_CACHE`（及 `PIP_CACHE_DIR`）重定向到同实例本地盘**（如 `/root/.cache/modelscope`，关机清空但不影响已落持久盘的目标文件），使缓存与目标**不叠加在 100GB 持久盘上**，从根上规避峰值爆盘**。G4 实测确认 `/mnt/workspace` 上是否复制，据此定默认策略；state 已按目标文件记录完成态，缓存清理/丢失不影响续传；
9. **磁盘预算总口径（区分峰值/稳定占用）**：持久盘**稳定占用** = Profile 模型体积 + ComfyUI 本体 ~3GB + venv（魔搭复用 `system-site-packages` 后通常 <1GB，非复用按 ~3GB）+ 已清理缓存(0)；**下载峰值占用** = 稳定占用 + 缓存（缓存落持久盘则 +~1×模型，落本地盘则 +0）。预检按"峰值占用"校验持久盘，但**当缓存已重定向到本地盘时，持久盘只需满足稳定占用**（否则 42GB 模型 + 100GB 盘按原口径会被误判不足而降档，实际稳定仅 ~46GB 足够）；低于阈值时按 §7.3 自动降档并 WARN，禁止在不足空间上硬写。

---

## 7. 脚本接口设计（全自动为默认）

### 7.1 文件清单（交付仓库布局）

```
veda-setup/                        # 交付仓库根（魔搭实例建议 /mnt/workspace/veda-setup）
├── install.sh                     # Linux/macOS 入口（LF 行尾；bash 3.2+ 兼容）
├── install.bat                    # Windows 入口（CRLF 行尾；纯 ASCII 内容）
├── setup_veda.py                  # 主编排驱动（跨平台，仅标准库）
├── assets_manifest.json           # 资产声明（含魔搭/HF 多源候选，UTF-8 无 BOM）
├── profiles.json                  # Profile 定义
├── README.md                      # 使用说明（人工唯一需要读的文件）
├── .gitattributes                 # 行尾固化：*.sh=LF，*.bat=CRLF，*.py=LF
└── lib/
    ├── phases.py                  # Phase 0–8 编排
    ├── assets.py                  # 下载/校验/源切换
    ├── diag.py                    # 日志、错误码、Doctor 诊断包
    └── platform.py                # 平台探测与适配（triton/venv/锁/启动方式）
```

> 注：`run/`（start/stop/status）与 `logs/` 属于**安装根**（§6），由安装过程生成，不在交付仓库内。入口脚本放在仓库根而非 `scripts/` 子目录，保证 `bash install.sh` / 双击 `install.bat` 的路径最短，且与 §14.1 使用说明一致。

### 7.2 命令行参数

| 参数 | 说明 | 默认值（全自动） |
|---|---|---|
| `--install-root PATH` | 安装根目录 | 自动探测（魔搭实例 → `/mnt/workspace/comfyui-veda`） |
| `--mode NAME` | 运行模式：`auto`/`install`/`launch` | **`auto`（默认）**：据 `install_completed` 自动判定——已安装仅启动（预检失败自动降级转安装段）、未安装则安装+启动；`install`：强制进入安装段（幂等增量，仅补缺失/过期项）后启动；`launch`：强制仅启动（跳过安装段，安装不完整报 E701） |
| `--only LIST`（v2.3.7） | 单环节执行：`p1`–`p8` 逗号分隔（如 `--only p5`、`--only p1,p3`；也接受 `5` 裸数字） | 无（默认执行安装段全流程）。给出后强制走安装段路径但**仅执行指定环节**（P0 预检始终前置，提供环境/档位/网络上下文）；`install_completed` **不变更**（部分执行不构成完整安装）；未含 `p8` 则不启动（等同 `--no-launch`）；与 `--mode launch` 同用记 WARN 并以 `--only` 为准 |
| `--profile NAME` | `minimal`/`standard`/`full`/`lowvram` | **自动**：按 P0 实测显存映射（<16GB→lowvram / 16–24GB→standard / >24GB→full），**设计层不假设任何 GPU**；磁盘不足自动降档并 WARN |
| `--source-priority LIST` | 下载源优先级（覆盖自动探测） | **auto（按环境自适应：魔搭 `modelscope,github-ghproxy,codeload`；通用 `modelscope,hf-mirror,huggingface`，E5/E6）** |
| `--skip-models` | 跳过主模型资产（只搭环境） | false |
| `--accept-license` | 接受社区 License | **true（自动接受并记日志；含地域限制提示，F18）** |
| `--accept-third-party` | 允许下载社区第三方量化权重（lowvram 档） | false |
| `--local-code` | 代码+venv 放本地盘（模型/state 留持久盘），加速启动 | **false（魔搭环境强制 OFF；仅 T2/T3 可选，E3）** |
| `--keep-cache` | 下载校验后保留 `MODELSCOPE_CACHE`（默认清理以省磁盘） | false（E1） |
| `--listen` | ComfyUI 绑定 `0.0.0.0`（默认仅 `127.0.0.1`，经 IDE 端口预览/隧道访问） | false |
| `--interactive` | 退回人工确认模式 | false |
| `--dry-run` | 只打印动作不落盘 | false |
| `--resume` | 显式从失败处续跑（与幂等重跑等价，保留供脚本化调用） | false |
| `--deep-verify` | 资产全量哈希校验 | false（默认存在性+大小+快速指纹） |
| `--smoke-test` | 无头启动 ComfyUI 探测 `/object_info` | false |
| `--no-launch` | 装完不启动 ComfyUI | false |
| `--doctor` | 只生成诊断包后退出（无安装根也可运行） | false |
| `--log-dir PATH` | 日志目录 | `<安装根>/logs` |

### 7.3 全自动默认行为表（零人工的保障）

| 决策点 | 默认行为 | 人工模式（--interactive） |
|---|---|---|
| License | 自动接受，日志记录 ACCEPT_LICENSE 事件（附 URL 与关键限制） | 展示摘要等待输入 |
| 安装根已存在同类安装 | 自动采用并增量升级 | 询问 |
| 安装根存在**外来** ComfyUI（非本脚本、非 git） | 报错 E601 并给出一行修复建议（换 `--install-root`） | 询问 |
| 下载体积大 | 直接开始，日志记录总体积 | 先询问 |
| 下载中断/额度耗尽 | 下次重跑基于 per-asset state 续传，已完成文件不重下 | — |
| 节点目录已存在 | 备份为 `*.bak-<ts>` 后升级 | 询问 |
| Profile 超磁盘 | 自动降到可容纳的最大档，WARN | 询问 |
| 浏览器启动 | 启动并尝试打开（无桌面则跳过） | 询问 |
| 运行模式（是否已安装） | `auto`：读 `install_completed` 自动判定——已安装→仅启动（跳过 P1–P7），未安装→安装+启动；`launch` 模式但 `install_completed=false` 直接 E701 退出 | `launch` 模式强制仅启动；`install` 模式强制走安装段（幂等增量，仅补缺失项） |

### 7.4 退出码（供自动化/CI 使用）

`0` 成功；非 0 为对应错误码（见 §11.2），主流程失败时**必输出**：错误码、失败 Phase、日志路径、诊断包路径。

### 7.5 跨平台兼容性检查表（多操作系统的落地约束）

| 项 | Windows | Linux | macOS | 处理方式 |
|---|---|---|---|---|
| 入口调用 | 双击 `install.bat` | `bash install.sh` | `bash install.sh` | 统一用 `bash install.sh` 调用，**不依赖可执行权限位**（zip 上传/部分同步工具会丢 +x） |
| 入口内 Python 探测 | 依次 `py -3` → `python` → 提示 `winget install Python.Python.3.12` 后退出 | `python3` → `python` | `python3` → `python` | 入口只负责找到解释器，找到后一律交 `setup_veda.py` |
| 行尾（**硬伤项**） | `.bat`=CRLF | `.sh`=LF | `.sh`=LF | `.gitattributes` 固化；CRLF 的 `.sh` 在 Linux/macOS 报 `bad interpreter`，直接失败 |
| 控制台编码 | 驱动内 `sys.stdout.reconfigure(encoding="utf-8")`（Python 3.7+），`.bat` 内容纯 ASCII 避免 CP936 乱码 | UTF-8 原生 | UTF-8 原生 | 中文提示在三平台一致 |
| venv 解释器路径 | `<root>/venv/Scripts/python.exe` | `<root>/venv/bin/python` | 同 Linux | 一律绝对路径调用，**不依赖 activate 脚本**（activate 三平台语法不同） |
| 文件锁 | msvcrt / O_EXCL lockfile | `fcntl.flock` | 同 Linux | `lib/platform.py` 分平台实现（§6 规则 6） |
| Triton / 稀疏内核 | `triton-windows`（装后复验 torch） | 原生 Triton（需 gcc/JIT） | MLX 后端 | §8 P2 |
| 运行脚本 | `run/*.bat` | `run/*.sh` | `run/*.sh` | P8 按平台生成对应后缀 |
| 路径长度 | MAX_PATH 260 风险（默认根在深目录时） | 无 | 无 | Windows 默认根保持短路径；超长时提示开启 Win32 长路径或换 `--install-root` |
| 换行/编辑器 | 文档与脚本统一 LF/CRLF 策略 | — | — | 交付前 CI 检查 `.sh` 无 CRLF |

---

## 8. 执行流程（Phase 0–8 + M0 运行模式判定）

```
 M0 运行模式判定（读 .setup-state.json 的 install_completed + 目录存在性）
        │
        ├─ auto 且 已安装 ─────────────► 轻量预检 ─► P8 启动            （跳过 P1–P7，秒级进入）
        ├─ auto 且 未安装 ─────────────► P0 → P1 → P2 → P3 → P4 → P5 → P6 → P7 → P8
        ├─ --mode install（强制安装段）► P0 → P1 → P2 → P3 → P4 → P5 → P6 → P7 → P8
        └─ --mode launch（强制仅启动）─► 轻量预检 ─► P8 启动            （install_completed=false → E701）
```

**概念拆分**：
- **安装段** = P0 预检 → P1 ComfyUI → P2 依赖 → P3 节点 → P4 预测器 → P5 主模型 → P6 模板 → P7 验收。全部成功后写 `install_completed=true`（见 §6 规则3），随后进入启动段。
- **启动段** = P8 启动（写 `run/` 脚本 + 后台拉起 ComfyUI + 就绪探测）。启动段不依赖安装段是否刚跑完——只要环境已装好即可运行，故已安装实例可独立进入。
- **幂等粒度不变**：安装段内每项仍按 per-asset state 跳过（§6 规则5），故「未安装但曾被中断」重跑只补缺失部分，不会全量重来。

### M0 运行模式判定（v2.3.5 新增，位于并发锁之后、安装段之前）

- 加载 state（`ctx.state.load()`），读取 `install_completed`、`install_profile`、`manifest_sha1`；
- **`is_installed()` 判定（真实、无网络假设，纯文件系统 + state）**：
  1. `state.install_completed == true`；
  2. `state.manifest_sha1 == 当前 manifest_sha1`（manifest 变更视为安装元数据过期 → 重新进入安装段，**幂等增量**：per-asset state 仍生效，仅下载新增/变更资产，不重下已完成项）；
  3. `ComfyUI/.git` 存在（本脚本安装的 git 仓库，排除外来 ComfyUI）；
  4. `ComfyUI/custom_nodes/Veda-on-ComfyUI/.git` 存在；
  - 四项全满足 → 判定「已安装」；否则「未安装」；
- **模式映射**：
  - `--mode install` → 强制走安装段 + 启动段（无视 `is_installed`）；
  - `--mode launch` → 强制仅启动段；若 `is_installed()` 为 False → 立即 `E701` 退出（提示先用 `auto`/`install`）；
  - `--mode auto`（默认）→ `is_installed()` 为真 → 仅启动段（其轻量预检若失败，**自动降级转安装段**而非报错，见「轻量预检」节）；为假 → 安装段 + 启动段；
- **`--only` 单环节执行（v2.3.7）**：给出 `--only pN[,pN]` 时跳过 M0 分流、强制走安装段路径，但仅执行指定环节（P0 预检始终前置以提供环境/档位/网络上下文）；`install_completed` **不复位也不置位**（部分执行不构成完整安装，避免半成品被 M0 误判）；未含 `p8` 则不启动；与 `--mode launch` 同用记 WARN 并以 `--only` 为准。用途：单环节重试（如 `--only p5` 补模型）、单环节重装（如 `--only p3` 重装节点）；
- **`--no-launch` 交互**：安装段末尾的 P8 仍受 `--no-launch` 控制（装完不启动）；`launch` 模式若同时带 `--no-launch` 视为无意义，记 WARN 并跳过启动（不报错，等同「仅确认已安装」）；
- **`install_completed` 复位（v2.3.6 补充）**：进入安装段（`run_install`）时先将 `install_completed` 复位为 `false` 并落盘；安装段全部成功后再置 `true`——防止重装/更新半途失败时旧标志残留，导致下次 auto 误判「已安装」直接启动缺资产的实例；
- **日志与可观测**：M0 必打印一条 `MODE_DECISION` 事件（`install` / `launch-only`），便于用户确认当前走的是哪条路径。

### P0 预检
- 平台/架构识别；Python 探测（Windows 依次 `py -3`/`python`，其余 `python3`/`python`；要求 3.10–3.12）；git 存在性（缺失不阻塞，走 tarball 安装路径）；
- GPU：`nvidia-smi` → 名称/显存/计算能力。**"环境不满足但可询问继续"的环节（v2.3.9/v2.3.10）统一走通用 `_confirm_continue`**：dry-run→继续；`--yes`→继续；TTY→询问 `[y/N]`（默认不继续）；非交互无 `--yes`→打印说明后干净退出（返回 0，不抛错）。适用于：① **无 NVIDIA GPU**（E103，沿用 fallback 档 `standard`）；② **GPU 算力 < SM80**（E102，Veda Triton INT8 后端需 SM80+，低算力可能无法出片）；③ **安装根存在非本脚本建立的 ComfyUI**（E601，覆盖继续；`--force` 等同 `--yes` 直接继续）。具体：
  - 无 GPU：自动档（`--profile auto`）未检测到 NVIDIA GPU 时交互询问 `仍继续安装（将使用 standard 档）？[y/N]`；用户确认或 `--yes` 后沿用 fallback 档 `standard` 继续（Veda 需 GPU 才能运行，CPU 模式不可用，无 GPU 安装仅用于准备环境/预下载，运行时仍需 GPU）；显式 `--profile X` 跳过询问直接采用指定档位。
  - 算力不足（< SM80）：GPU 已检测但 `compute_cap_ok` 为 False 时同样交互询问是否继续（v2.3.10 起；此前 `compute_cap_ok` 为死代码、E102 从未触发，与文档"报 E102 退出"不符，现已修正为可询问）。
  - 外来 ComfyUI：安装根已存在且非本脚本 `.git` 仓库时交互询问是否覆盖继续（v2.3.10 起；此前直接 E601 退出）。
  - **默认 Profile 严格按实测显存映射**（<16GB→lowvram、16–24GB→standard、>24GB→full），**设计层不假设任何 GPU 型号**；A10 仅是魔搭免费实例的实测硬件，standard 是其在 24GB 下的预期档位，最终由运行时确认（F14/G6）；
- 磁盘：**可用空间 vs（模型体积 + ComfyUI ~3GB + venv ~3GB + 缓存预留 ~1.5×模型，见 §6 规则8/9）**，按真实 `Content-Length` 汇总；低于阈值自动降档并 WARN（E1）；
- 网络探测：`modelscope.cn`、`hf-mirror.com`、`huggingface.co`、`github.com`、`gh-proxy.com` 等，生成《源可用性表》；**魔搭实例内 huggingface.co 直连受限（F17+2026-10-12 实测），但 hf-mirror 实测可达**——模型源链为 `modelscope → hf-mirror → gh-proxy 代理 HF(末位)`，代码源链为 `github 直连(首位尝试) → gh-proxy → codeload(末位)`；通用环境为 `modelscope → hf-mirror → huggingface`（E5/E6，详见 §9.2）；
- **魔搭实例识别**：存在 `/mnt/workspace` 且镜像预装 torch → 标记 `env=modelscope`，后续复用系统 torch，并**强制 `--local-code` 关闭**（E3）；
- 存在**未完成的安装**（`install_completed` 未置位，如曾被中断）→ 读 state 文件（校验 `platform/profile/manifest_sha1`），输出《恢复计划》（将跳过/将执行的清单）；**已安装实例（`install_completed=true`）已在 M0 分流至仅启动路径，不会进入 P0**；
- **并发锁（含 stale 检测，在驱动入口获取——先于 M0 模式判定，安装/启动两路径共用）**：创建 `.setup.lock`（含 PID）；若已存在且持有者存活 → E603 退出；若持有者已死 → 自动释放后继续（§6 规则6）；输出《环境报告》。

### P1 ComfyUI 本体
- state 记录版本；**manifest 锁定已知可用 commit（可复现，防上游破坏）**；已安装且 ≥0.38.0 → 跳过；已安装但更低 → `git fetch` + `git checkout <commit>` 升级；未安装/升级 → 按锁定 commit 获取（**SHA 不能 `--depth 1 --branch` 后 checkout——浅克隆不含历史 commit 会失败**），正确做法：`git clone --filter=blob:none --no-checkout <url> <dest>` → `git -C <dest> fetch origin <commit>` → `git -C <dest> checkout <commit>`（blobless，不全量历史）；**GitHub 经 gh-proxy 代理**：临时 `git config url."https://gh-proxy.com/https://github.com/".insteadOf "https://github.com/"`，仅局部/本次会话，用完还原，禁止 Token（E6）；git 失败降级 tarball：`https://<proxy>/https://github.com/<repo>/archive/<commit>.tar.gz`（**注意用 commit SHA 而非分支名，保证可复现**，多代理候选依次尝试）；两者都失败才 E201；
- 非 git 的外来安装 → E601。
- **阶段级快速跳过（v2.3.7，重试提速）**：`ComfyUI/.git` 存在且 `git rev-parse HEAD` == manifest 锁定 commit → P1 整体直接跳过（不 fetch、不探测代理——魔搭内 git 直连必失败，重复 fetch 是纯浪费）；仅 HEAD 不匹配或目录缺失时才进入获取/升级链。`_git_clone` 对已存在仓库同样先比对 HEAD，匹配即返回。
- **tarball 幂等标记（v2.3.8）**：tarball 兜底安装的仓库**无 `.git`**（HEAD 比对不适用，且 `is_installed()` 按设计要求 `.git` 存在）——旧实现下重试会 rmtree 重下整个仓库。修正：tarball 解压后在仓库根写入标记文件 `.tarball-ref`（内容=ref/commit）；`download_repo` 快速跳过条件扩展为「`.git`+HEAD 匹配 **或** `.tarball-ref` 匹配」，命中即零网络跳过（commit 变更时标记不匹配，自然走重下链）。

### P2 运行依赖
- venv：魔搭/系统有 torch 时用 `--system-site-packages` 复用（**省去数 GB torch 下载**），否则按 CUDA 版本装官方 wheel；torch 升级只进 venv，不动系统包；
- `pip install -r ComfyUI/requirements.txt`；`PIP_CACHE_DIR` 指向持久缓存；
- **pip 失败回退**：任一步 pip 失败或导致 `import torch` 不可用 → 丢弃该 venv、重建干净 venv 并装 CUDA torch 后重试（记 WARN + REBUILD_VENV 事件）；
- `modelscope` 包缺失则 `pip install modelscope`（下载主源）；
- Windows 特判：`import triton` 失败 → `pip install triton-windows`，**装后必须复验 `import torch`**，失败则回滚该包并记录手动命令（避坑 F8；triton 与 torch 版本强耦合）；
- Linux 容器内 triton JIT 需 gcc，缺失则 WARN 并尝试 `apt`（无权限时记录手动命令）；
- macOS：装 MLX；
- ComfyUI-Manager 为 **best-effort**：失败仅 WARN，不影响主流程。

### P3 Veda 节点 + MiniMax-H3 加载器（E2，关键）
- **Veda 加速节点**：clone/升级 `custom_nodes/Veda-on-ComfyUI`（git → tarball 降级链同 P1，commit 处理同 §8 P1），**按 manifest 锁定 commit checkout**（§9.1 可复现性）；装节点自带 requirements（若有）；Manager/Registry 方式作为最终兜底；全部失败才 E301。**重试快速跳过（v2.3.7）**：节点 `.git` 存在且 HEAD == 锁定 commit → 跳过 clone；节点依赖（requirements/FA4）以节点目录内标记文件（`.veda-req-done`，内容=锁定 commit）判重——标记匹配即跳过 pip 安装，commit 变更后自动重装；
- **FA4 运行时（best-effort，可选增强）**：若节点含 `install_fa4.py`（FA4 后端，CUDA13 CuTe DSL / macOS MLX），best-effort 运行其自测；默认 Triton INT8 后端由节点自带内核 + Triton 提供（SM80+ 即可），FA4 仅为可选加速，失败不阻断主流程；
- **MiniMax-H3 模型加载器（仅装 Veda 无法出片，必须处理）**：**已实测确认走 ComfyUI 原生加载器（G3）**——ComfyUI 自 v0.38.0 起原生支持 MiniMax-H3（官方 PR `Comfy-Org/ComfyUI#15224`，2026-08-03 合并，新增 4 个原生节点 + 6 个官方工作流模板），故**不 clone 任何第三方加载器节点**（原 `Comfy-Org/ComfyUI-MiniMaxH3` 经 G2 核实不存在 404）。本阶段**显式校验原生加载器节点存在**（`object_info` 中节点 id 含 `minimax`/`h3`，不区分大小写）；加载器与 Veda 串联（MODEL: 加载器 out → Veda in → Veda out → sampler）由 P6 模板保证。

### P4 预测器权重（×2 ≈ 525MB：T2VA/R2VA 各 ≈262MB，魔搭实测）
- 从 manifest 取 T2VA/FL2VA（+R2VA，随 Profile）；
- 源链：遵循 §9.2 auto 自适应链（魔搭内 `modelscope → hf-mirror（实测可达，优先兜底）→ gh-proxy 代理的 HF（末位 best-effort）`；通用 `modelscope → hf-mirror → HF`），文件以魔搭页当前为准（G1；两个预测器仓库与文件名已于 2026-10-12 经魔搭 API 实测确认存在，t2va=275,415,648B、r2va=275,415,752B）；
- **下载落点规则（v2.3.7 修正，实测发现的 E401 根因）**：CLI `--local_dir` 会按仓库内路径展开——`include` 型源（仓库内带子目录，如 Comfy-Org 的 `diffusion_models/xxx`）用 `--local_dir <root>/ComfyUI/models`；`file` 型源（仓库根裸文件名，如 Veda 预测器）**必须**用 `--local_dir <target 的父目录>`（`models/veda/`），否则文件落在 `models/` 根而校验目标在子目录，下载成功也判失败；
- **错位文件回收（v2.3.7）**：检测到旧版本脚本下载的错位文件（`models/<裸文件名>` 存在而 target 缺失且体积≥min_bytes、头部可解析）→ 直接迁移到正确位置并校验，不重新下载；
- 校验：≥250MB + safetensors 头部 JSON 可解析（防 HTML 错误页/截断）；
- 完成写 state（含 sha256 前 64KB 快速指纹，供跨实例比对）。

### P5 MiniMax-H3 主模型资产（`--skip-models` 可跳过）
- **License 合规**：`--accept-license` 默认 true，本阶段开始前记录 `ACCEPT_LICENSE` 事件（含 License URL 与地域限制 F18，详见 §10），`--interactive` 下改为人工确认；
- manifest 按 Profile 生成清单 → 汇总真实体积（`Content-Length`）→ 日志记录 → 开始；
- 主源：`modelscope download --model Comfy-Org/MiniMax-H3 --include "<子目录>/<文件>" --local_dir <root>/ComfyUI/models`（目录结构以魔搭页为准，F9/G1；`--include` 子目录 glob 语义在 M3 用小文件实测确认）；
- 单文件失败按 §9.2 auto 链降级（魔搭内次选 **hf-mirror（实测可达，优先兜底）**，gh-proxy 代理 HF 仅末位 best-effort；通用环境走 hf-mirror → HF）；3 次重试 + 指数退避；下载落点与错位回收规则同 §8 P4（v2.3.7）；
- **免费额度可能中途耗尽**：所有下载可随时中断，重跑时基于 per-asset state 续传（modelscope/hf CLI 均支持断点），已完成文件不重下；
- 全部落位后输出《资产核对表》（文件/目标目录/大小/校验结果）。

**standard Profile 清单（魔搭 A10 24GB 预期档位；文件大小均按魔搭官方实测校准，非估算：DiT 19.5 + TE 14.6 + video_vae 4.85 + audio_vae 0.56 + lora 1.82 + veda 0.26 ≈ 41.6GB，另 ComfyUI/venv 各 ~3GB；VRAM 占用为新实例首次实测项 G6，由 P0 运行时确认，不足自动降档）**：

| 文件 | 目标目录 |
|---|---|
| `minimax_h3_fl2va_pruned_int8_convrot.safetensors` | `models/diffusion_models/` |
| `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | `models/text_encoders/` |
| `minimax_h3_audio_vae_fp32.safetensors` | `models/vae/` |
| `minimax_h3_video_vae_fp16.safetensors`（或 `_int8_convrot`，以魔搭页当前文件为准） | `models/vae/` |
| `minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors` | `models/loras/` |

### P6 工作流模板
- **Veda 节点自带官方模板（已实测确认 G3）**：节点 `example_workflows/` 提供 'Veda MiniMax H3 T2VA' / 'Veda MiniMax H3 R2VA'，经 ComfyUI *Workflow → Browse Templates → Veda-on-ComfyUI* 加载；本阶段 (a) 校验节点 `example_workflows/` 存在；(b) best-effort 将所需模板拷贝到 `ComfyUI/user/default/workflows/` 便于直接加载；**不依赖任何外部模板仓库下载**（原 `Comfy-Org/workflow_templates` 兜底未证实，已废弃，避免假设）；
- 扫描模板：同链同时存在 Veda 与 ComfyUI 原生 Model Sparse Attention → WARN + 修复指引（避坑 F7）。

### P7 验收（自动）
1. ComfyUI 版本 ≥0.38.0；2. **MiniMax-H3 加载器节点可见**（原生，ComfyUI ≥0.38.0 内建，E2）+ Veda 节点可导入、目录就位；3. 预测器校验通过；4. `torch.cuda` 可用且算力 ≥8.0、`triton` 可导入；5. Profile 资产全绿；6. `--smoke-test`：无头启动，轮询 `/object_info` 出现 `VedaSparseAttention`（及原生加载器节点）后关闭；**首跑豁免全模型加载**（冷启动需加载 ~42GB 模型常 >180s），超时放宽至 **600s**，超时/失败仅记 WARN + 单独错误码，标注"首跑豁免/待人工复核"，**不阻断整体验收结论**。
- 结果写入 state 与日志；任一项失败 → 错误码 + 诊断包。

### P8 启动
- 生成 `run/start.sh|stop.sh|status.sh`（Windows 为 `.bat`）；`start.sh` 以 **`nohup ... >> run/comfyui.log 2>&1 &` 后台启动**并写 pid 文件，`status.sh` 查健康、`stop.sh` 停止（E4：避免前台启动卡死安装命令，安装命令可立即返回）；
- **已运行守护（v2.3.8）**：启动前先探测 `http://127.0.0.1:8188/object_info`——**已就绪则跳过启动**（记 `P8/skip`），避免已安装实例重复执行命令时二次拉起实例（端口冲突，且旧实例响应就绪探测会掩盖新实例启动失败）；`run/start.sh` 同样内置该守护（UP 即退出）；
- **会话脱离（v2.3.8）**：安装期首启（P8 直接 Popen）POSIX 侧使用 `start_new_session=True`（等价 nohup，脱离控制组），**终端会话关闭/SIGHUP 不再杀死 ComfyUI**；Windows 侧重定向 stdout 到 `run/comfyui.log`（DETACHED_PROCESS 无控制台，直接 print 可能异常）；日志统一**追加**（不截断，保留历史便于排错）；
- Windows `stop.bat` 用 PowerShell 按 `CommandLine` 匹配 `main.py` 精确停止（v2.3.8；旧 `taskkill /FI WINDOWTITLE` 匹配不到无窗口的 python.exe）；
- **默认绑定 `127.0.0.1`**，通过 Web IDE 端口预览/隧道访问（云实例暴露面最小化）；仅 `--listen` 时绑 `0.0.0.0:8188`；
- 后台启动后轮询 `/object_info` 探测就绪（复用 P7 逻辑，期望 `VedaSparseAttention` 节点可见，超时 120s）：**v2.3.11 起——就绪后打印访问 URL/端口与《使用摘要》；若进程已退出（崩溃，如 CUDA/OOM/依赖错误）则抛 E701 并就地打印 `run/comfyui.log` 尾部（不再谎报成功）；若进程存活但 120s 未就绪（大模型加载较慢）则诚实标记"未就绪"，`final_summary` 不再宣称可访问，提示用 `run/status.sh` 复查**；《使用摘要》含模板入口、默认参数（稀疏度 90%/90%）、避坑提醒、`Ctrl+B` 对比法、日志与诊断包位置。

### 轻量预检（launch 路径专属，非安装段）

launch 路径（auto 判定为已安装，或显式 `--mode launch`）**不跑 P0 的磁盘预算/网络探测/降级/外部 ComfyUI 检测**，仅做启动必需的最小准备，保证「已安装实例秒级进入 ComfyUI」：
- `platform.detect_platform()` 识别环境（魔搭复用系统 torch 的路径常量）；
- 解析 venv 解释器路径（`platform.venv_python(root)`）并校验存在且可执行（`-c "import sys"` 探测）；**launch 路径是只读的——绝不创建/重建 venv**；
- 校验 `ComfyUI/main.py` 存在；
- **v2.3.11 CUDA 早期信号**：虽不跑完整 P7，轻量预检仍用 venv 探测 `torch.cuda.is_available()`——**不可用则 WARN 提前预警**（Veda 需 GPU 才能运行，无 GPU 时 ComfyUI 启动即崩溃、端口永不开放），不阻断（尊重显式 launch 意图）；
- 其余（GPU 档位、网络源链、磁盘降档）全部跳过——这些只在安装段才有意义。

**预检失败分流（关键，v2.3.5 自检补充）**：venv 缺失/失效（如新实例镜像 Python 版本变化）、`main.py` 缺失等——
- **auto 模式**：**自动降级转安装段**（记 `MODE_DECISION=install`，`fallback_reason=launch-preflight-failed`）——安装段幂等增量：P2 检测 venv 失效仅重建 venv（pip 缓存命中，分钟级），模型/ComfyUI 已就绪则跳过，**不重下 42GB 资产**；转完后继续 P8 启动，全程无人工；
- **`--mode launch`（显式强制）**：不做降级，立即 `E701` 退出（尊重用户显式意图，避免静默副作用）；

> **独立启动入口**：因 `run/start.sh|stop.sh|status.sh` 在安装段即已生成，已安装实例也可**直接** `bash <root>/run/start.sh` 启动，完全不经过 `setup_veda.py`；但 `install.sh`/`setup_veda.py`（默认 `auto`）会在「已安装」时自动只做启动段，二者等价。

---

## 9. 资产清单与下载策略

### 9.1 manifest 结构（声明式，示例）

```json
{
  "schema": "veda-assets/v1",
  "assets": [
    {
      "id": "veda-predictor-t2va",
      "title": "Veda 预测器（T2VA/FL2VA）",
      "target": "ComfyUI/models/veda/minimax_h3_t2va_veda_8nfe_600step_preview_fp8.safetensors",
      "min_bytes": 262144000,
      "sources": [
        {"type": "modelscope", "repo": "Veda-Sparse/Minimax-H3-T2VA-Veda-8NFE-600Step-Preview", "file": "minimax_h3_t2va_veda_8nfe_600step_preview_fp8.safetensors"},
        {"type": "hf", "repo": "Veda-Sparse/Minimax-H3-T2VA-Veda-8NFE-600Step-Preview",
         "file": "minimax_h3_t2va_veda_8nfe_600step_preview_fp8.safetensors", "endpoint": "https://hf-mirror.com"}
      ]
    },
    {
      "id": "h3-dit-int8",
      "title": "MiniMax-H3 DiT（int8_convrot，~19.5GB，魔搭实测）",
      "target": "ComfyUI/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
      "sources": [
        {"type": "modelscope", "repo": "Comfy-Org/MiniMax-H3", "include": "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"},
        {"type": "hf", "repo": "Comfy-Org/MiniMax-H3", "include": "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors", "endpoint": "https://hf-mirror.com"}
      ]
    }
  ]
}
```

> **可复现性（v2.3.2）**：repo 类资产（ComfyUI 本体、`Veda-on-ComfyUI`）在 manifest 中**锁定 `commit`/`tag`** 字段（已回填 G1/G2 实测的真实 commit：`ComfyUI=7f7fd918…`、`Veda-on-ComfyUI=67e4eb55…`，均为 2026-10），脚本按锁定值 clone/checkout，避免上游 `main` 破坏性更新导致"昨天能装今天崩"。MiniMax-H3 加载器走 ComfyUI 原生（无第三方节点仓库）。模型权重类资产以文件名 + `min_bytes`（已按魔搭官方清单校准为真实大小安全下界）+ 头部校验保证。

### 9.2 源切换逻辑

```
候选链 = --source-priority（默认 auto）过滤 + P0 可达性探测结果（E5/E6）
  魔搭实例·代码源:  github(gh-proxy.com) → github(ghproxy.net) → github(gitclone.com) → codeload(末位, 同样走代理)
                   （github 直连不稳定——2026-10-12 实测时而通时而不通，直连仅作首位尝试）
  魔搭实例·模型源:  modelscope → hf-mirror（实测可达，优先兜底）→ gh-proxy 代理 HF（末位 best-effort）
                   （huggingface.co 直连实测不可达；hf-mirror 实测可达——v2.3.7 依实测修正，旧版"排除 hf-mirror"作废）
  通用环境·模型源:  modelscope → hf-mirror（探测可达时）→ huggingface 直连（末位）
                   （v2.3.8：hf-mirror 不可达时跳过之直接 HF 直连——海外用户镜像反而绕路；hf 直连始终作末位兜底）
state 以资产为最小粒度记录（见 §6 规则5），中断恢复精确到单个文件
对每个资产：按候选链逐源尝试 → 成功即校验 → 通过则落盘写 state；全部失败 → E401 + 诊断包
下载落点：include 型源 --local_dir=<root>/ComfyUI/models；file 型源 --local_dir=<target 父目录>（§8 P4）
```

- ModelScope 源：`modelscope download --model <repo> [--include <glob>] [<file>] --local_dir <局部目录>`；缓存默认重定向到 `<root>/.cache/modelscope`，**当持久盘容量紧张时改指本地盘 `/root/.cache/modelscope`（§6 规则8b）以规避峰值爆盘**；下载校验通过后默认清理该缓存（`--keep-cache` 可关）；
- HF 源：`HF_ENDPOINT=<mirror> huggingface-cli download <repo> <file> --local-dir <局部目录>`（通用环境使用）；
- **GitHub 类源（节点/ComfyUI 本体兜底）统一走可达代理**：`git` 经 `git config url."https://gh-proxy.com/https://github.com/".insteadOf ...`（局部，用完还原，禁止 Token）；tarball 用 `https://gh-proxy.com/https://github.com/<repo>/archive/<ref>.tar.gz`；多代理候选（gh-proxy.com / ghproxy.net / gitclone.com，均 2026-10-12 实测可达；mirror.ghproxy.com 已废弃）依次探测；
- 所有下载直落目标目录，不落 `/tmp` 再拷贝（魔搭实例 `/tmp` 停机即失，且省磁盘）；
- 校验三件套：存在性 → 大小下限 → safetensors 头部 JSON 可解析；`--deep-verify` 加 sha256 全量。

---

## 10. 避坑映射（文章 + 仓库 + 魔搭实例实践 + 自检补充）

| 坑 | 脚本处理 |
|---|---|
| Windows "no sparse kernel works"（F8） | P2 强制 `import triton`，失败自动 `triton-windows`，装后复验 torch |
| Linux 容器 triton JIT 缺 gcc | P0/P2 检测 `cc`，缺失 WARN 并记录手动安装命令 |
| Veda 与原生 Model Sparse Attention 混串（F7） | P6 扫模板 WARN + 修复指引 |
| Mem Eff Sage Attention Patch 顺序（F7） | 使用摘要固化说明 |
| 预测器缺失节点直接抛错（F4） | P4 预下载+校验 |
| predictor 为 preview，仅覆盖少量分辨率/时长 | 验收建议固定 1344×768/124 帧；`nearest trained size` 含义写入使用摘要 |
| 8–12GB 显存 | `lowvram` Profile（第三方量化，`--accept-third-party` 显式开启） |
| **魔搭实例 `/tmp`、`/root`、`/home` 停机清空（F11）** | 安装根/venv/缓存/日志全部落 `/mnt/workspace`；P0 检测到非持久路径写入即 WARN |
| **NAS（/mnt/workspace）IO 较慢** | 记录各阶段耗时，超阈值 WARN；`--local-code` 仅 T2/T3（本地持久盘）可选加速，**魔搭环境强制 OFF（§6 规则7）**；M4 实测 T2/T3 本地盘加速效果 |
| **镜像 Python 版本变化致 venv 失效** | P0 检测 venv 解释器版本，失效则仅重建 venv（缓存命中，分钟级） |
| License 合规（F13） | P5 前记录 ACCEPT_LICENSE 事件（附 URL 与关键限制；默认自动接受，`--interactive` 人工确认） |
| **GitHub 不可达（F15）** | P1/P3 clone 失败降级 tarball 镜像解压；Manager best-effort |
| **pip 触及其预装包致 torch/triton 损坏** | venv 失败重建干净 venv；triton-windows 装后复验 `import torch`，失败回滚 |
| **standard 档 VRAM 占用（G6，首实例实测，非设计假设）** | P0 按实测显存映射档位 + 自动降档兜底；新实例首次跑通后回填 A10 实测数据与默认 Profile（F14） |
| **免费额度耗尽致下载中断（F16）** | per-asset state + 断点续传，重跑续传 |
| **两次安装并发** | `.setup.lock`，冲突即 E603 |
| **`.sh` 带 CRLF 行尾（跨平台硬伤）** | `.gitattributes` 固化行尾；交付前 CI 检查；`bash install.sh` 调用方式不依赖 +x |
| **ModelScope 缓存与目标重复占满 100GB 持久盘（E1）** | P0 磁盘预算计入 `MODELSCOPE_CACHE`+`PIP_CACHE_DIR`（~1.5× 模型）；下载校验后默认清理 `.cache/modelscope`（`--keep-cache` 关闭）；M3 实测 `/mnt/workspace` 是否复制 |
| **只装 Veda 缺 MiniMax-H3 加载器，工作流缺节点（E2）** | P3 显式校验 ComfyUI 原生加载器（已移除第三方节点兜底 `ComfyUI-MiniMaxH3`——G2 核实该仓库不存在 404）；P7 增加"加载器节点可见"验收 |
| **`--local-code` 与"免重装"在魔搭自相矛盾（E3）** | 魔搭环境（检测到 `/mnt/workspace`）强制 `--local-code` OFF，全部内容落持久盘；仅 T2/T3 可选 |
| **P8 前台启动卡死安装命令（E4）** | P8 改为 `nohup ... &` 后台启动 + 就绪探测，安装命令立即返回；提供 run/start·stop·status |
| **魔搭 Notebook 外网受限含 huggingface（F17/E5，2026-10-12 实测）** | huggingface.co 直连不可达、hf-mirror 可达：模型源链 `modelscope → hf-mirror → gh-proxy 代理 HF(末位)`；代码源链 `github 直连(首位尝试) → gh-proxy → codeload` |
| **GitHub 兜底用 codeload 同样被墙（E6）** | GitHub 类源统一走可达代理（gh-proxy.com 等，多候选+探测），禁止带 Token；用完还原 git insteadOf |
| **MiniMax H3 许可地域排除（F18）** | P5 自动接受 License 时日志/README 提示：不适用于美/欧/英/韩用户；商业使用受限；中国个人非商业无碍 |
| **并发锁崩溃后永久死锁** | `.setup.lock` 含 PID/stale 检测，持有者已死自动释放（§6 规则6） |

---

## 11. 日志与诊断设计

### 11.1 结构化日志

- 每次运行：`<root>/logs/setup_YYYYMMDD_HHMMSS.jsonl`，每行一条 JSON：
  `{"ts","level","phase","step","event","detail","duration_ms","error_code"}`
- 控制台同步人类可读输出（INFO 简略、WARN/ERROR 高亮）；
- 关键事件强制记录：`ACCEPT_LICENSE`、`SOURCE_FALLBACK`（源切换）、`PHASE_SKIP`（幂等跳过）、`ASSET_OK`（含大小/指纹）、`ASSET_FAIL`、`REBUILD_VENV`、`VERIFY_FAIL`、`ROOT_ADOPT`；
- 轮转：保留最近 10 次 setup 日志 + 5 个诊断包，超出自动清（**删除模式严格限定 `setup_*.jsonl` / `diagnose_*.tar.gz`，只清 logs 目录内自有文件**）。

### 11.2 错误码注册表

| 码 | 含义 | 人工动作 |
|---|---|---|
| 0 | 成功 | — |
| E101–E106 | 预检：OS/Python 不符、GPU SM<80、无 GPU、磁盘不足、安装根不可写、网络全不可达 | 按日志《环境报告》处理 |
| E201–E204 | ComfyUI/依赖：clone 及 tarball 均失败、版本<0.38.0、pip 失败、venv 重建失败 | 看日志网络/源可用性表 |
| E301 | 节点安装失败（所有方式） | 检查 github 可达性 |
| E401–E403 | 资产：所有源失败、校验失败、License 未接受（交互模式） | 换源/重试；E403 加 `--accept-license` |
| E501–E504 | 验收：版本/资产缺失、triton 不可用、object_info 无 Veda 节点 | 带诊断包提 issue |
| E601–E603 | 安装根：存在外来 ComfyUI、无写权限、并发锁冲突 | 换 `--install-root` / 等待或终止另一安装 |
| E701 | 运行模式或启动失败：① **显式 `--mode launch` 但安装不完整**（`install_completed=false`，或轻量预检发现 venv 缺失/失效、`ComfyUI/main.py` 不存在等）；② **P8 启动崩溃**（v2.3.11 起：后台拉起 ComfyUI 后 120s 内进程已退出，如 CUDA 不可用/OOM/依赖错误）。**auto 模式不报此码**——安装不完整场景会自动降级转安装段；但 P8 崩溃在任意模式均报 E701（端口永不开放，无法继续） | ① 去掉 `--mode launch`（默认 auto 会自动安装/修复+启动）或改 `--mode install`；② 查看 `run/comfyui.log` 尾部定位崩溃原因（多为无 GPU/显存不足/OOM），确认 `nvidia-smi` 与 `python -c "import torch; print(torch.cuda.is_available())"`；环境无 GPU 时无法运行 Veda，需更换 GPU 实例 |

### 11.3 `--Doctor` 一键诊断包

`logs/diagnose_<ts>.tar.gz`（Windows 为 zip），内容：
- 环境快照：`uname -a`、`nvidia-smi`、`python -V`、`pip freeze`、`df -h`；
- **版本快照**：`modelscope`/`pip` 版本、ComfyUI 与 `Veda-on-ComfyUI` 的当前 commit（便于复现与排障）；
- **网络/egress 探测表**：各源（modelscope/hf-mirror/huggingface/github/gh-proxy）延迟与可达性，及**生效的源链结论**（魔搭内 hf 是否被限，E5/F17）；
- 本次运行配置（参数+生效的源链+Profile+`--local-code` 实际值）；
- `.setup-state.json` 全文；
- 最近 2000 行日志；
- 安装根文件树（路径/大小/快速指纹）；
- ComfyUI 运行日志尾部（如有）。

失败时主流程**自动**生成诊断包并在结尾打印路径，用户提 issue 只需附这一个文件。无安装根时 `--doctor` 退化为纯环境快照，仍可运行。

---

## 12. 安全设计

1. **写范围收敛**：只写安装根、日志目录、pip/modelscope 缓存目录；不碰系统目录、不改 PATH、不写注册表；
2. **零删除用户数据**：不删除任何用户资产/模型/安装产物；同名节点目录先备份 `*.bak-<ts>`；git 仅 fast-forward；仅 `logs/` 下脚本自有日志/诊断包按 §11.1 轮转清理，不触碰其他目录；
3. **外来安装保护**：安装根存在非本脚本建立的 ComfyUI → E601 拒绝覆盖；
4. **License 透明**：自动接受也落日志（时间、版本、URL、文件清单），可审计；
5. **第三方量化隔离**：仅 `--accept-third-party` 时下载社区量化包，并提示来源与风险；
6. **网络暴露最小化**：云实例默认绑 `127.0.0.1`，经 IDE 端口预览/隧道访问；`--listen` 才绑 `0.0.0.0`。

---

## 13. 验收标准（DoD）

- [ ] 魔搭 Notebook 全新实例：一条命令装到可生成（首次，含 ~42GB 下载）；
- [ ] **关机 → 新开实例 → 同一条命令 → ≤5 分钟恢复可用（资产零重下）**；
- [ ] **安装/启动分离验收**：首次实例跑同一条命令完成安装+启动；在已安装实例**再次跑同一条命令**（`auto` 默认），日志出现 `MODE_DECISION=launch-only`，**跳过 P1–P7 安装段**（不重下模型、不重建 venv），直接进入 P8 启动；`--mode launch` 在已安装实例等价；`--mode launch` 在未安装实例立即 `E701` 退出；**新实例镜像 Python 变化致 venv 失效时，auto 自动降级转安装段仅重建 venv**（日志 `fallback_reason=launch-preflight-failed`，模型零重下），随后照常启动；
- [ ] Windows 本地：双击 `install.bat` 无人值守完成；
- [ ] **Linux + Windows 入口必须各跑通一次**：`bash install.sh`（Linux）、`install.bat`（Windows）均能到达 P7 验收；macOS 为实验性（T4），允许 MLX 分支差异，作为尝鲜验证即可；
- [ ] ComfyUI 版本 ≥0.38.0，**节点面板同时出现 MiniMax-H3 加载器节点与 `Veda Sparse Attention (MiniMax H3)`**（E2）；
- [ ] 模板可加载无缺失弹窗；节点输出 `Veda done · Triton INT8 (SMxxx)`，Attention computed ≈10%；
- [ ] `Ctrl+B` 旁路同 Seed 对比画质可接受；
- [ ] **G6 输出 A10 24GB 实测报告**：VRAM 占用、生成耗时、Veda 生效确认（预期 `Triton INT8 (SM86)`）；若 standard 档 VRAM 不足，运行时自动降档并回填默认 Profile（F14/G6）；
- [ ] **磁盘不溢出验收**：首次安装后 `/mnt/workspace` 占用 < 100GB，且 modelscope 缓存已清理（E1）；魔搭环境 `--local-code` 实测为 OFF、venv 在持久盘（E3）；
- [ ] 任意失败场景：退出码非 0 + 错误码 + 诊断包路径，凭诊断包可定位。

---

## 14. 使用说明（人工侧，只有一条命令）

### 14.1 魔搭 Notebook（推荐）

**方式 A · bootstrap（需真实托管地址）**：将 `veda-setup/` 推到一个可访问的仓库（GitHub / GitLab / 魔搭 / OSS）后，用一行命令拉取并执行（把下方 `<真实仓库地址>` 换成你的仓库）：

```bash
curl -fsSL https://<真实仓库地址>/bootstrap.sh | bash
# 之后每次新开实例，只需：
bash /mnt/workspace/veda-setup/install.sh   # 资产已持久化，分钟级恢复
```

> 注：bootstrap 依赖一个真实可访问的托管地址（当前未提供），不内置任何假定地址；若暂无托管地址，请直接用方式 B。

**方式 B · 手动（当前可用）**：
```bash
cd /mnt/workspace
git clone <脚本仓库地址> veda-setup        # 或通过 Web IDE 上传 veda-setup/ 目录
bash veda-setup/install.sh
```

安装完成后 ComfyUI 在**后台启动**（P8），终端立即返回并给出访问地址（Web IDE 端口预览/隧道，默认 8188）。选模板 "Veda MiniMax H3 T2VA" 直接出片。

> **安装 / 启动自动分流（v2.3.5）**：`install.sh` / `setup_veda.py` 默认 `auto` 模式——**首次实例**自动执行「安装段 P0–P7 + 启动 P8」；**已安装实例**再次跑同一命令，会读取 `install_completed` 自动判定为「仅启动」，跳过 P1–P7 安装段（不重下模型、不重建 venv），秒级进入 ComfyUI。无需人工区分「安装」还是「启动」。
> 显式控制：强制进入安装段（幂等增量）`python setup_veda.py --mode install`；强制仅启动 `python setup_veda.py --mode launch`（未完成会报 E701）；已安装实例也可直接 `bash /mnt/workspace/comfyui-veda/run/start.sh` 纯启动。
> 许可提示：MiniMax H3 社区许可**不适用于美国/欧盟/英国/韩国用户**（F18），请确认你的使用地域与用途合规。

### 14.2 Windows 本地

双击 `install.bat`（或 `python setup_veda.py`），全程无交互，结束后自动打开浏览器。机器无 Python 时按提示安装（`winget install Python.Python.3.12`）后重跑即可；git 缺失不阻塞（自动走 tarball 安装）。

### 14.3 出问题怎么办

1. 看终端最后三行（错误码 + 日志路径 + 诊断包路径）；
2. 确认运行模式：日志首行会有 `MODE_DECISION=install` 或 `launch-only`，据此判断是否走了安装段；
3. `python setup_veda.py --doctor` 重新生成诊断包；
4. 带 `logs/diagnose_*.tar.gz` 提 issue。
> 若误用 `--mode launch` 报 E701：说明安装不完整，去掉该参数（默认 `auto` 会自动安装/修复并启动）或改用 `--mode install`。

---

## 15. 实施计划

| 里程碑 | 内容 | 产出 |
|---|---|---|
| M1 | 驱动骨架：`setup_veda.py` + `platform.py` + `diag.py`（JSONL 日志/错误码/Doctor）+ P0 预检（含魔搭实例识别、并发锁、三平台 Python 探测） | 干净实例上输出正确《环境报告》与《恢复计划》 |
| M2 | P1–P3：ComfyUI 安装/升级（git→tarball 降级）、venv（system-site-packages 复用 + 失败重建）、Veda 节点 | "带 Veda 节点的 ComfyUI" |
| M3 | P4–P5：manifest + 下载引擎（魔搭 CLI 优先、源切换、三件套校验、per-asset 续传；`--include` glob 语义小文件实测） | 预测器 + 主模型资产就位 |
| M4 | P6–P8：模板、验收、start/stop 脚本；**魔搭实例端到端实测，第一项为 A10 24GB VRAM 占用（G6 运行时实测，非设计假设）**，含端口访问方式、NAS 速度、实跑提速数据 | 一键可用 + 实测报告（回填默认 Profile） |
| M5 | Windows 实测（triton-windows 路径、无 Python/git 场景）+ README/使用说明 + 幂等恢复演练（关机重开）+ `.gitattributes` 行尾 CI 检查 | 全平台验收通过 |
| M6 | **v2.3.5 安装/启动分离（✅ 已实现，随 v2.3.6 落地）**：M0 运行模式判定 + `is_installed()`（4 项判定）+ state 顶层 `install_completed`/`install_profile` 字段（含进入安装段时复位）+ launch 专用轻量预检（只读）+ `--mode {auto,install,launch}` 参数 + E701 + `MODE_DECISION` 日志事件；拆分 `run_install`（P1–P7 + 写 install_completed）与 P8 启动段 | 已安装实例重复执行同一命令输出 `MODE_DECISION=launch-only` 并直接启动（DoD 新增验收项），不重下模型、不重建 venv |

---

## 16. 开发前必验清单（Gates，进入 M1 前须确认）

| Gate | 待确认项 | 若不成立的影响 |
|---|---|---|
| G1 | ✅ **已实测确认**：`Comfy-Org/MiniMax-H3` 在 **ModelScope** 存在（`Downloads≈386万`、`License=minimax-h3-community-license-agreement`、`StorageSize≈525GB` 含多档量化；目录含 `diffusion_models/ text_encoders/ vae/ loras/ embeddings/ model_patches/`，文件名与 HF 一致，F9 已采信） | 主模型可走魔搭源（第一源），速度/稳定性最佳 |
| G2 | ✅ **已实测确认**：Veda 预测器在 **ModelScope** 有镜像——`Veda-Sparse/Minimax-H3-T2VA-Veda-8NFE-600Step-Preview`（~262MB）、`Veda-Sparse/Minimax-H3-R2VA-Veda-Preview`（~262MB），`License=minimax-h3-community` | P4 主源走魔搭，最佳 |
| G3 | ✅ **已实测确认**：ComfyUI 原生 MiniMax-H3 加载器存在——官方 PR `Comfy-Org/ComfyUI#15224`（2026-08-03 合并）新增 4 个原生节点 + 6 个官方模板；pin `7f7fd918…`（2026-10）≥0.38.0 含该支持；第三方 `ComfyUI-MiniMaxH3` 节点经核实**不存在（GitHub 404）**，不依赖 | 否则工作流缺节点、无法出片 |
| G4 | `modelscope download --local_dir` 在 `/mnt/workspace` 上**是否复制缓存**（E1） | 决定磁盘预算与缓存清理策略 |
| G5 | 魔搭实例内 `gh-proxy.com`/`gitclone.com` 是否可达（E6） | 决定 GitHub 兜底是否可用 |
| G6 | A10 24GB 实测能否跑 `standard` 档（G6，运行时实测，非设计假设；F14 已改为运行时显存映射） | 决定默认 Profile 与降级路径 |
| G7 | 魔搭免费实例**本地盘（/root 或系统盘）可用容量**是否 ≥ 最大 Profile 模型体积（~19.5GB，即 DiT int8；非此前估算的 55GB）；连同 G4 的"是否确为 copy"共同决定：缓存能否重定向到本地盘避险，以及 standard 档在 100GB 持久盘是否真正可装（若本地盘不足且确为 copy，可能仅 `lowvram` 档可装） | 决定缓存重定向策略与默认可装档位 |

> 其余开放项：
> - **脚本仓库地址（影响方式 A bootstrap，不影响方式 B）**：`bootstrap.sh` 需要一个可访问的托管地址（GitHub/GitLab/魔搭/OSS），由你提供；方式 B（Web IDE 上传目录 / git clone）无需该地址即可使用，当前即可交付。
> - **G4–G7 为运行时探测项，非设计假设**：缓存复制行为（G4）、gh-proxy 在魔搭可达性（G5）、A10 实测 VRAM（G6）、魔搭本地盘容量（G7）均无法在本地预验，但设计已做自适应兜底——缓存重定向本地盘（§6 规则8b）、源链按 P0 探测（§9.2）、显存映射自动降档（P0），故不依赖这些项的预设结论。
> - **首次下载体积与档位**：魔搭持久盘 100GB；standard 档 ≈42GB（已按魔搭官方实测校准，非估算）；VRAM 可行性 G6 实测，不足自动降档。

---

## 附录：修订记录

| 版本 | 变更 |
|---|---|
| v1.0 | 首版：Windows-first、PowerShell 实现（§5.1 文件清单为纯 Windows 技术栈，不支持多平台，已被 v2.0 取代） |
| v2.0 | 按 5 条新需求重构：单一 Python 驱动、魔搭持久化部署、全自动默认、魔搭 CLI 优先、JSONL 日志+错误码+Doctor |
| v2.1 | 自检修订：per-asset state 与 manifest 失效机制、并发锁（E603）、clone/pip 降级链与回滚、Windows 无 Python/git 处置、`--local-code` 与 NAS 性能权衡、默认 `127.0.0.1` 绑定、VRAM 假设 A1 与降级路径、额度耗尽续传、smoke-test 超时、P6 模板探测 |
| v2.2 | 文件清单修正（回答"§5.1/§7.1 是否正确、是否支持多平台"）：入口脚本移到仓库根并与使用说明对齐、新增 README.md 与 .gitattributes；新增 §7.5 跨平台兼容性检查表（入口调用方式、Python 启动器探测、**.sh=LF/.bat=CRLF 行尾固化**、Windows 控制台 UTF-8、venv 解释器绝对路径、分平台文件锁、运行脚本后缀、MAX_PATH 处理） |
| v2.3 | 合并《设计方案评审与优化建议》全部修正：① 新增外部事实 F17（魔搭外网受限含 hf）/F18（许可地域排除）/F19（GitHub 代理）；② 缓存可能翻倍撑爆 100GB → 磁盘预算计入缓存+校验后清理（E1）；③ 缺 MiniMax-H3 加载器 → P3 显式装/校验（E2）；④ `--local-code` 魔搭强制 OFF（E3）；⑤ P8 后台启动避免卡死（E4）；⑥ 源链自适应+排除被墙 hf-mirror（E5）；⑦ GitHub 兜底走可达代理而非 codeload（E6）；⑧ 锁 stale PID 释放；⑨ manifest 锁 commit 可复现；⑩ smoke 首跑豁免+长超时；⑪ 新增 `--keep-cache`、bootstrap 单行命令、许可地域提示、诊断包增强；G1–G6 前置验证清单 |
| v2.3.1 | 自查修正（无脚本开发，仅文档）：修复 §9.1 孤立 ```` ``` ```` 致 §9.2 标题被吞的渲染 bug；修正悬空引用 §4/O1、笔误 E6/V5、hfmirror→hf-mirror；统一 F9↔G1（魔搭镜像"已存在"降为"疑似待 G1 确认"）；对齐 P4/P5 降级链与魔搭排除 HF（§9.2 auto 链）；将 License 记录步骤明确写入 P5；修正 `--local-code` 仅 T2/T3 可选（魔搭强制 OFF）表述；澄清 §12 删除逻辑仅限 logs 自有文件；macOS 由 DoD 必跑降为实验性尝鲜；P3 补 manifest commit 锁定 checkout；**深化 E1**：指出"仅清理缓存救不了峰值爆盘"，新增缓存重定向本地盘（§6 规则8b）+ 峰值/稳定占用分口径（规则9），新增 G7（本地盘容量核查） |
| v2.3.2 | **G1/G2 实测确认 + manifest 修正**（脚本 M1–M5 已实现）：① G1 ✅ `Comfy-Org/MiniMax-H3` 在魔搭存在（Downloads≈386万、License=minimax-h3-community、含 diffusion_models/text_encoders/vae/loras 等子目录，文件名与 HF 一致）；② G2 ✅ Veda 预测器两镜像在魔搭存在（T2VA/R2VA 各 ~262MB）；③ G2 复核 `Comfy-Org/ComfyUI-MiniMaxH3` **不存在（404）** → 废弃原 P3 方案 B，统一走 ComfyUI 原生加载器；④ 回填真实 commit（ComfyUI=`7f7fd918…`、Veda=`67e4eb55…`，2026-10）锁定可复现；⑤ **修正全部 `min_bytes` 与体量**：按魔搭官方清单校准，原两处下限竟大于真实文件（TE nvfp4_awq 14.6GB、audio_vae 0.56GB）会导致正确下载被误判失败，现已纠正；standard 模型体量由估算 55GB 修正为 ~42GB（DiT 19.5+TE 14.6+video_vae 4.85+audio_vae 0.56+lora 1.82+veda 0.26）；同步更新 F9、P3、P7、§9.1、§16 G1–G7、DoD 与本文修订记录 |
| v2.3.3 | **设计假设清零（v2.3.2 基础上的实测审计）**：① 实测确认 F11（魔搭 100GB/A10/root/预装，官方文档 `modelscope.cn/docs/notebook/intro`）、F17（魔搭外网受限含 huggingface，官方文档原文）、F18（MiniMax H3 许可地域排除 EU/UK/ROK/USA，LICENSE 原文）、F19（gh-proxy.com/ghproxy.net/gitclone.com/hf-mirror 实测可达，mirror.ghproxy.com 不可达已废弃）、G3（ComfyUI 原生 MiniMax-H3 加载器 PR#15224 实测存在）；② 修正 §9.1 占位 `<魔搭镜像ID·实施时核实>` → 真实 `Veda-Sparse/Minimax-H3-T2VA-Veda-8NFE-600Step-Preview`；③ 修正 commit 下载机制（SHA 不能 `--depth 1` 后 checkout，改 blobless clone+fetch SHA / tarball 用 SHA 归档）；④ P6 改为 Veda 节点自带模板（废弃未证实的 Comfy-Org/workflow_templates 兜底）；⑤ P3 补 FA4 best-effort；⑥ 移除设计层"假设 A1"（默认档位改 P0 实测显存映射，非预设 A10）；⑦ bootstrap 不再用假地址；⑧ 新增 F20（lowvram 第三方量化源未证实，预留开关）；同步更新 F9/F11/F14/F17/F18/F19、P0/P1/P3/P4/P5/P6、§9.2、§10、§13、§14.1、§16 G3 与本文修订记录 |
| v2.3.4 | **代码实现对齐 v2.3.3（无新增假设）**：① `lib/assets.py` 移除失效代理 `mirror.ghproxy.com`，候选链改为 gh-proxy.com/ghproxy.net/gitclone.com（§9.2）；② 修复 `_git_clone` 的 SHA 克隆 bug：`blobless clone --no-checkout` → `git fetch origin <sha>` → `git checkout <sha>`，分支/标签仍走轻量 `--depth 1 --branch`（§8 P1）；③ 修复 `_tarball`：SHA 用 `archive/<sha>.tar.gz`、分支用 `archive/refs/heads/<ref>.tar.gz`，`download_repo` 透传 SHA（不再误回退 `main`）；④ `lib/phases.py`：P3 补节点自带 `requirements.txt` + `install_fa4.py` best-effort（均非致命）；P6 显式拷贝 Veda 节点 `example_workflows/*.json` 到 `user/default/workflows/`；P7 清理已失效的 `ComfyUI-MiniMaxH3` 死引用，仅报告原生加载器依赖。Python 3.10 下 py_compile 全通过，20 项逻辑自测（SHA 检测/代理链/克隆序列/解包 URL/phases 导入）全绿 |
| v2.3.5 | **安装/启动分离与已安装自动判定（设计层，代码待 M6 实现）**：① 新增 M0 运行模式判定（§8）：`--mode {auto,install,launch}`，默认 `auto`——据 `is_installed()` 自动分流，已安装实例仅启动（跳过 P1–P7 安装段），新实例自动安装+启动；② `is_installed()` 四项判定（纯文件系统 + state，无网络/外部假设）：`install_completed=true` ∧ `manifest_sha1` 一致 ∧ `ComfyUI/.git` 存在 ∧ `Veda-on-ComfyUI/.git` 存在；③ state 顶层新增 `install_completed`/`install_profile`，安装段（P0–P7）全部成功后写入（§6 规则3）；④ 新增 launch 专用轻量预检（只读：env + venv 路径校验 + `main.py` 存在性，不跑磁盘/网络/降档、不创建 venv）；**预检失败分流：auto 自动降级转安装段（幂等增量，仅重建失效 venv，不重下模型），仅显式 `--mode launch` 报 E701**——覆盖"新实例镜像 Python 变化致 venv 失效"场景（F11/§10），不让用户被 E701 卡死；⑤ 新增 E701 错误码（仅限显式 launch）与 `MODE_DECISION` 日志事件（含 `fallback_reason`）；⑥ 同步更新 §1.2 目标、§4 映射表、§5 架构图、§7.2 参数表、§7.3 默认行为表、§8 流程图与 P0 措辞（锁在驱动入口先于 M0 获取）、§11.2、§13 DoD（新增安装/启动分离验收项，含 venv 失效降级用例）、§14.1/14.3 使用说明、§15 新增 M6 里程碑。**自检修正**：`--mode install` 统一表述为「强制进入安装段（幂等增量）」而非「强制重装」（避免误导全量重下）；轻量预检由 `ensure_venv`（缺失时会创建，有写副作用）改为 `venv_python()` 仅解析路径+校验 |
| v2.3.6 | **M6 代码实现落地（按 v2.3.5 设计，无新增外部假设）**：① `lib/diag.py`：注册 E701；`State` 默认数据含 `install_completed=false`/`install_profile=null`（损坏重置分支同步）；② `lib/phases.py`：新增 `is_installed()`（四项判定）、`decide_mode()`（M0 模式映射 + `MODE_DECISION` 事件 + 显式 launch 未安装即 E701）、`preflight_launch()`（只读轻量预检：env + `venv_python()` 路径校验 + `main.py` 存在性，失败抛 701，auto 由上层降级）；`run_phases` 重构为 `run_install`（P1–P7 + **进入时复位 `install_completed=false`** + 成功后置 `true` 并记 `install_profile`），P8 启动段独立由驱动调用；③ `setup_veda.py`：新增 `--mode {auto,install,launch}`；主流程重构为「锁 → M0 判定 → launch 路径（轻量预检→P8，auto 预检失败自动降级转安装段并记 `fallback_reason`）／install 路径（P0→run_install→P8）」；dry-run 输出模式判定；清理 `version_snapshot` 中已废弃的 `ComfyUI-MiniMaxH3` 死引用；④ 使用说明/问题排错/README 同步 `--mode`、E701 与自动分流说明。本地校验：py_compile 全通过；自测覆盖 is_installed 四分支/decide_mode 全映射/701 抛出/复位落盘/argparse 语义 |
| v2.3.7 | **首次魔搭真实实例实测反馈修复（E401 复盘，代码随本版实现）**：实测环境：魔搭 A10 24GB / `/mnt/workspace` / P0–P4 一次通过（档位 standard、缓存重定向、stale 锁自动释放均符合设计）。**根因修正**：① P5 `veda-predictor-t2va` E401——`download_model_asset` 把 `file` 型源（Veda 预测器，仓库根裸文件名）统一下载到 `--local_dir models/` 根，而 target 在 `models/veda/` 子目录，**下载成功但落错位置致校验失败**（魔搭 API 实测两预测器仓库/文件均存在：t2va=275,415,648B、r2va=275,415,752B，min_bytes 下界有效，排除大小问题）；修正为按源类型分派 local_dir（§8 P4 落点规则）+ 错位文件迁移回收（不重下）；② hf 兜底端点错误——魔搭内被改写为 gh-proxy 代理 HF（不代理 HF，必失败）而 hf-mirror 实测可达；修正源链为 `modelscope → hf-mirror → gh-proxy HF(末位)`（§9.2/P0/P4/P5 同步）；③ 重试无阶段级跳过——P1 旧跳过条件 `not meta_changed("","","")` 恒 False（state 真实元数据 vs 空串永不等），每次重试重复 fetch（魔搭 git 直连必败，纯浪费）；P3 依赖每次重装；修正为 commit 匹配快速跳过（P1/P3/`_git_clone` 三层）+ `.veda-req-done` 标记判重；④ 新增 `--only pN[,pN]` 单环节执行（§7.2/M0：P0 前置、install_completed 不变更、未含 p8 不启动）。 |
| v2.3.8 | **全量代码自检修正（代码随本版实现）**：P0 级——① **P8 重复启动缺陷**：已安装实例在 ComfyUI 运行中重跑命令会二次拉起实例（端口冲突，旧实例响应 `/object_info` 掩盖新实例失败）→ P8/start.sh 增加「已就绪即跳过」守护；② **安装期首启无会话脱离**：POSIX 直接 Popen 无 `start_new_session`，魔搭 Web IDE 终端关闭（SIGHUP）会杀死 ComfyUI → 修复；Windows DETACHED_PROCESS 未重定向 stdout → 重定向到日志；③ `profiles.json` 残留过期假设（"约 55GB；VRAM 可行性假设 A1"——v2.3.3 已删；minimal 档描述"--skip-models 默认生效"不实）→ 清理。P1 级——④ 通用环境模型源链缺 HF 直连末位兜底（manifest endpoint 全为 hf-mirror，镜像故障无后路）→ 补齐（§9.2）；⑤ P0 磁盘降档 `set_meta` 后缀不一致（丢 `/modelscope`）→ 元数据每次运行来回翻转 → 修正；⑥ `--only p3/p7/p8` 在 venv 缺失时 FileNotFoundError 裸崩 → 守卫（P3/P7 warn 跳过对应检查，P8 报 E701）；⑦ P2 二次 pip 失败冒泡为"未预期错误"（无错误码）→ 补 E203；⑧ Windows `stop.bat` 的 `taskkill /FI WINDOWTITLE` 匹配不到无窗口 python.exe → 改 PowerShell 按 CommandLine 匹配 `main.py`；⑨ tarball 兜底安装无 `.git`，重试每次 rmtree 重下仓库 → `.tarball-ref` 幂等标记（§8 P1）；⑩ dry-run 的 P0 `makedirs(pip_cache)` 落盘副作用 → 跳过。P2 级（记录不改）——`comfyui.log` 由截断改追加；`_file_tree` 的 `/.git` 分隔符 Windows 失效（仅诊断包）；Windows 退出码 701→189 截断（平台限制，排错文档已注明）。 |
| v2.3.9 | **无 GPU 交互继续（代码随本版实现）**：P0 GPU 预检此前在无 NVIDIA GPU 时直接 `PhaseError(103)` 退出，用户期望"先问是否继续"。修正（`lib/phases.py` `phase0_preflight` + 新增 `_confirm_no_gpu_continue`；`setup_veda.py` 新增 `-y/--yes`；`lib/diag.py` E103 改为"非致命、交互询问"注释）：① 自动档（`--profile auto`）无 GPU 不再直接退出——终端 TTY 下交互询问 `仍继续安装（将使用 standard 档）？[y/N]`（默认不继续）；② `--yes`（`-y`）直接继续（无交互自动确认，含非交互场景）；③ 非交互且无 `--yes` → 打印可操作说明后**干净退出（返回 0）**，不抛错误、不挂起；④ 用户确认/`--yes` 后沿用 fallback 档 `standard` 继续安装（Veda 需 GPU 才能运行，无 GPU 安装仅用于准备环境/预下载）；⑤ 显式 `--profile X` 仍跳过询问直接采用指定档位；⑥ dry-run 仅展示计划（不询问、不中止）。错误码 E103 不再作为致命错误抛出（用户拒绝为干净退出，非 E103 失败）。本地校验：py_compile 全过；自测 `_confirm_no_gpu_continue` 6 分支（dry-run/--yes/非交互/TTY-y/TTY-n/TTY-空）全绿；`phase0_preflight` 无 GPU 集成测试 5 场景（非交互中止/--yes→standard/显式 minimal/ TTY-y→standard /TTY-n 中止）全绿。 |
| v2.3.10 | **同类"可询问跳过"环节扩展 + 自检修正（代码随本版实现）**：① **E102 算力不足改交互询问**：`platform.compute_cap_ok` 此前为死代码（从未调用，与文档"报 E102 退出"不符），修正——GPU 已检测但 `compute_cap_ok` 为 False（< SM80）时同样走通用 `_confirm_continue`，不再直接退出；② **E601 外来 ComfyUI 改交互询问**：安装根存在且非本脚本 `.git` 仓库时由硬 `raise E601` 改为 `_confirm_continue`（默认不继续；`--yes`/`--force` 直接继续）；③ **抽出通用 `_confirm_continue(ctx, prompt)`** 助手，GPU/E102/E601 三处复用（原 `_confirm_no_gpu_continue` 移除）；④ **`_file_tree` 诊断包修正（v2.3.8 已知 P2）**：跨平台用 `os.sep→"/` 归一化再做片段匹配（Windows 原生 `\\` 旧实现漏匹配，且 venv 片段误写为 `/.venv` 永不匹配，导致 venv/cache 内成千上万文件被全量列出、诊断包膨胀）；⑤ **P7 验收区分"无 CUDA"与"triton 不可用"**，避免无 GPU 安装时误报 triton；⑥ 清理 P0 `return ctx` 死返回值。`Ctx.user_aborted` 统一承载 GPU/E102/E601 的"用户拒绝"干净退出（驱动层返回 0）。本地校验：py_compile 全过；自测 `_confirm_continue` 6 分支全绿；`phase0_preflight` 集成测试扩展至 7 场景（无 GPU 非交互中止 / 无 GPU `--yes`→standard / 显式 minimal / TTY-y→standard / TTY-n 中止 / **E102 非交互中止 / E102 `--yes` 继续**）/ **E601 非交互中止 / E601 `--force` 继续**）。 |
| v2.3.11 | **P8 启动诚实化修复（代码随本版实现，回应"端口未开放却报成功"实测反馈）**：根因——`phase8_launch` 在 120s 就绪探测超时后仅 WARN 并**无条件 `log.ok("P8","done","启动完成")` + `final_summary` 打印可访问 URL**，且启动子进程句柄被丢弃，无法区分"崩溃"与"仍在加载"，真实启动错误（如 CUDA/OOM）被埋进 `run/comfyui.log` 无人可见。修正（`lib/phases.py`）：① 保留 Popen 句柄，`_poll_object_info` 超时后检查 `proc.poll()`——进程已退出则 `log.error` 抛 **E701** 并就地打印 `run/comfyui.log` 末尾 50 行（顶层捕获器输出"失败：E701" + 诊断包），**不再谎报成功**；进程存活但超时则诚实标记 `ctx.launch_ready=False`，`final_summary` 不再宣称可访问、提示 `run/status.sh` 复查；② 新增 `Ctx.launch_ready` 标记（ready/skip→True，超时未就绪→False）；③ **launch 路径轻量预检补 `torch.cuda` 可用性早期信号**（`preflight_launch` 用 venv 探测 `torch.cuda.is_available()`，不可用即 WARN 提前预警，不阻断）——因 launch 跳过了 P7，原无任何 GPU 检查，无 GPU 容器会直冲启动并崩溃。本地校验：py_compile 全过；自测新增 P8 三场景（就绪→`launch_ready=True` / 崩溃→抛 E701 / 存活超时→`launch_ready=False` 不抛错）+ `final_summary` 未就绪诚实分支，全绿（测试总 33 项）。 |
| v2.3.12 | **同类"谎报/不完整"缺陷自检与修复（代码随本版实现，回应"下载不完整/启动不完整/超时不规范"复查）**：① **`final_summary` 三态化（修复"未启动却谎报可访问"）**——旧实现仅区分 `launch_ready=False`（诚实），`launch_ready=None`（本次根本未启动：如 `--no-launch` / 仅 `--only` 安装环节 / `--mode launch --no-launch`）会落入"一键搭建完成"分支并打印 `http://127.0.0.1:8188` 可访问 URL（实际无服务在跑）。修正为三态：`True`→带 access 字段；`False`→诚实提示未就绪；`None`→明确"未启动 ComfyUI"并指引 `run/start.sh`，**绝不输出 false access**；② **`verify_safetensors_header` 增强数据段完整性（修复"下载不完整却判定成功"）**——旧仅校验头部 JSON，头部合法但**数据段被截断**（下载中断、文件大小仍 ≥ min_bytes）仍会通过并被记 `ok`。现按张量 `data_offsets` 最大值要求 `8 + 头长 + 数据段长 == 实际文件大小`，截断即判失败；③ **`download_model_asset` 回收 + 清理**——新增"文件存在且校验通过但 state 未记录（上一轮下载成功却在写 state 前中断）"直接复用免重复下载；每次源尝试前删除残留半截 `target`（避免 CLI 续传损坏文件、或误把残留当已下载校验通过）；④ **`download_repo`/`_git_clone` 损坏 `.git` 自动重建**——`.git` 存在但 `rev-parse HEAD` 读不出（克隆/检出中断）时 `rmtree` 后重新获取，而非放任 fetch/checkout 反复失败。本地校验：py_compile 全过；自测新增 [8] safetensors 数据段完整性（合法/截断/坏头）、[9] download 回收+清理（恢复完整文件 / 残次清理后二源成功）、[10] final_summary 三态（None 无 access / True 有 access / False warn），测试总 **43 项全绿**。 |
| v2.3.13 | **torch 运行时不兼容早暴露（代码随本版实现，回应实测 `AttributeError: module 'torch.library' has no attribute 'custom_op'`）**：根因——`comfy_kitchen`（被 ComfyUI 的 `comfy/quant_ops.py` 引入）需要 `torch.library.custom_op`，该 API 自 **torch>=2.0** 才有；而魔搭环境 venv 用 `system_site_packages` **复用镜像自带 torch**（脚本不自行安装 torch，`phases.py` P0 `ctx.system_site_packages = is_ms`），若镜像 torch < 2.0 则 `import comfy_kitchen` 阶段即 `AttributeError` 崩溃，且 v2.3.11 前的 CUDA 早期信号仅查 `torch.cuda.is_available()`（旧 torch 仍 import 成功、仅 cuda=False）无法拦截。修正（`lib/phases.py`）：① 新增 `_torch_compat_ok(venv_python)` 探针，执行 `import torch; print(__version__); print('custom_op' if hasattr(torch.library,'custom_op') else 'no_custom_op')`，解析 `hasattr` 结果（**精确整行匹配**，避免 `no_custom_op` 子串误判），返回 `(ok, detail)`；② **launch 预检 `preflight_launch` 在启动前调用**——torch 不兼容直接 `raise E701(f"torch 运行时不兼容，无法启动 ComfyUI：{detail}")`，顶层捕获器输出"失败：E701"+诊断包，给出可操作原因（"选用 torch>=2.0 的 GPU 镜像"）而非崩溃后 dump 原始 traceback；③ **install 验收 `phase7_verify` 同步探测**（仅 `log.warn`，与"无 CUDA"同级不硬阻断），明确预警"torch 运行时不兼容，ComfyUI 启动将失败"。本地校验：py_compile 全过；自测新增 `test_torch_compat`（torch 2.1.0→ok / 1.13.1→too_old 带"torch>=2.0" / import 失败→"torch 无法导入"），测试总 **46 项全绿**。 |
| v2.3.14 | **启动前置硬门禁（代码随本版实现，回应实测"安装成功却启动必崩、且 `install_completed=true` 致每次重跑都崩"循环）**：根因——v2.3.13 的 `_torch_compat_ok` 仅在 launch 预检（`preflight_launch`）与 install 验收（`phase7_verify`，仅 WARN）两处使用；auto 模式因 launch 预检 torch 不兼容而降级转**安装段**后，`phase8_launch` 仍**无条件 spawn** ComfyUI，在无 GPU / torch 不兼容容器上必崩（等 120s + dump traceback），且安装段已写 `install_completed=true`，导致后续每次 `install.sh` 都 `launch-only → 崩溃`。修正（`lib/phases.py` + `setup_veda.py`）：① 新增 `_launch_precheck(ctx)` 综合判定——venv 存在 + `_torch_compat_ok`（torch>=2.0 / `custom_op`）+ `torch.cuda.is_available()`，返回 `(ok, reason)`；② **`phase8_launch(ctx, strict=True)` 加启动前置门禁**——进入即 `_launch_precheck`，环境不满足则 `strict=True` 直接 `raise E701(reason)`（**绝不 spawn 必崩进程**、不等待 120s、不 dump traceback）；`strict=False`（auto/install 安装段顺带启动）环境不满足则 **跳过启动**（`log.warn` + `ctx.launch_skipped_reason=reason` + `launch_ready=None`），安装本身已完成、不崩溃、不谎报；③ 驱动层——`--mode install`/auto 降级安装路径 `phase8_launch(ctx, strict=False)`，`--mode launch`/`--only p8` 保持 `strict=True`；④ **`final_summary` 据 `launch_skipped_reason` 明确"未启动（环境不满足运行条件）"** 并指引换「带 NVIDIA GPU 且 torch>=2.0」镜像后 `run/start.sh` 启动；⑤ 同步让 `_torch_compat_ok` 探针显式 `import torch.library` 以防 `torch.library` 子模块未自动导入导致的误判。本地校验：py_compile 全过；自测新增 [11] `phase8_launch` 启动前置门禁（strict 抛 E701 且 Popen 未调用 / best-effort 跳过且不 spawn / 环境满足正常 spawn）+ [12] `_launch_precheck` 四环境判定（venv 缺失 / torch 不兼容 / 无 GPU / 全满足）+ `final_summary` 跳过诚实分支，测试总 **57 项全绿**。 |
