> 后续部署已切回原始 NVIDIA NVFP4，保留本次引擎融合。当前入口和验证见 [NVFP4 部署记录](NVFP4_DEPLOYMENT_020.md)。下文 FP4-Local 入口与结果是升级当时的历史记录，不代表现役默认配置。

# 上游 v0.2.0-beta 融合：2026-10-20

融合发布标签 `v0.2.0-beta`，提交 `3d1fb48090e8f2cefc22d7b5845fb7f5b71f33fa`。
应用/终端版本为 0.2.0-beta，引擎版本为 0.1.45.3；升级前本地提交为 `de77abb6`。
本次本机平台为 Windows、RTX 5090 32 GB、9950X3D、96 GB RAM、CUDA 13、MSVC 14.44。

## 入口与现役配置

| 入口 | 用途 | 地址 |
|---|---|---|
| `START-APP-SAFETENSORS.bat` | 上游新版应用界面的本地浏览器版；模型页选择 FP4-Local Native 或实验版 | localhost:8891 |
| `START-FP4LOCAL-PARALLEL.bat` | 稳定版 API / 原有网页 | localhost:8880 |
| `START-FP4LOCAL-NVIDIA-W4A4.bat` | 实验版 API / 原有网页 | localhost:8880 |

一次只启动一份模型。新界面加载时分配独立 API 端口并生成临时密钥；外部客户端
继续使用旧 8880 入口，或使用界面显示的服务地址与密钥。关闭浏览器标签不停止模型；
使用界面的 Unload / Quit Strata，或前台终端 Ctrl+C。

两份现役配置移至仓库根目录的 `strata-fp4local-native.json` 和
`strata-fp4local-w4a4-native.json`，符合上游发现规则，不再维护第二份同步副本。
原桌面 BAT 调用链已更新。配置包含本机绝对路径，其他电脑应修改路径。

当前引擎为 `build-native-engine/strata-safetensors-v020beta.exe`，SHA256：
`8ddc64bd35b38499f99235ac82434a2a8b2b2cf0c6170f86116ef2ebf51f97de`。
同一程序按环境开关选择稳定 FP32 专家 decode 或实验 W4A4 decode。

保留升级前未提交的本机设置：双路、262144 配置窗口、8192 prefill、BF16 KV、
1536 MiB 显存预留、GPU BF16 视觉（1024 tokens/图）、PCIe fraction 0、
自适应交换 4/96，以及 temperature 1.0 / top_p 0.95 / top_k 20。
`--spec 0` 和 `STRATA_BATCH_MTP=0` 保持原样，MTP 暂时关闭。
`STRATA_ROUTE_TAIL_SKIP=0`、`STRATA_COMMIT_BATCH=0` 继续关闭对应近似/提交优化。

## 融合内容与适配

- 引入桌面壳与应用源代码、终端 CLI、Control、签名更新/打包工具、服务端生命周期、
  排队/取消/监控及引擎 CUDA/CPU 更新，保留上游来源与许可证。
- 原始 safetensors、CSF/MXFP8、FP4-Local、锁定专家 RAM、W4A4/CUTLASS prefill、
  BF16 KV、原生槽复制及会话保护继续保留；主模型不转换为 GGUF。
- 上游 GDN 并行投影分支改走能识别 BF16 的分派，避免原始 dense 权重误入量化 MMVQ。
- Control 检查 HF tokenizer 和索引中的全部分片；`--mtp native` 不再被当作目录。
  原始权重计入共享存储盘点；索引检查不读取 tensor payload。
- 应用识别两份原生配置，优先推荐已准备好的稳定版，使用 `.venv-native`；
  性能模式保留原预留，避免套用 GGUF 的文件专家/部分常驻回退。
- 原生内存估算保留 GPU 热专家的完整 RAM 副本；80 GB RAM / 110 GB 合计是
  界面保守估算，不是新硬件验收。实际专家 arena 仍为 63.282 GiB，未包含其他开销。
- Windows 虚拟环境由实际 CPython 进程承载受管 host/server，并保留 venv 包路径，
  避免重定向器 PID 与 ready 文件 PID 不一致。身份校验继续保留。
- 源码版应用去掉空 API 密钥环境变量，使用私有配置中的实例密钥，兼容新版服务端
  对显式空密钥的检查；真实聊天调用不需要关闭认证。

新应用状态保存在忽略目录 `data/app-v020beta/`，与官方已安装应用独立。
本机编码代理复用已安装的 0.2.0-beta Harness；也可显式设置 `STRATA_HARNESS_DIR`。
新电脑构建应用 UI 需在 `apps/host` 执行 `npm ci`、`npm run build`。
Python 依赖见 `requirements-native.txt`，终端交互监控另见 `requirements-monitor.txt`。

## 验证

原始 JSON 和日志位于本机忽略目录 `work/upgrade-020beta/`。

- CUDA 原生引擎及选定测试目标构建通过。CUTLASS、槽提交、会话缓存/内存/快照
  5 个 CTest，以及 prefill 选择器 7 个 CTest 通过。
- BF16 attention oracle/graph 程序通过；独立 safetensors reader、布局、source guard、
  分段 PLE、可见边界及 FP4-Local dense plan 共 7 个 CTest 通过。
- 原始 HF tokenizer 对照 43 项、增量编码 30 项通过。
- 895 项服务端 unittest，20 项跳过；首次一个 Windows lifeline 进程登记项报错，
  在隔离 Windows PowerShell 模块环境后复跑该组 6 项通过（2 项跳过）。
- 新应用前端构建和 TypeScript 类型检查通过；原生 UI/内存估算/配置/解释器专项
  28 项通过。继承应用整套首次运行有 Windows 符号链接权限、临时文件清理及 shell
  环境失败；未将这套报告记为全绿。
- 稳定版：英文、中文、代码、算术、EOS 五组，与升级前程序生成的全部 token 相同。
  上限 64 tokens；不是全面质量或速度评估。
- 真实双路文本：8 个场景通过，含 greedy、seeded、惩罚参数、取消/恢复、续聊、
  输出上限、EOS 和 9043-token 长槽增长/复用旁路接纳。并发输出与各自单独运行一致。
  使用 16384 配置窗口、8192 prefill、BF16 KV、视觉、独立 KV、设备槽复制和自适应交换。
  记录 57 个多槽交换轮次、4707 次晋升，驻留后专家/MTP 源读取为 0。
- 真实图片：22 个请求、6 组并发通过，覆盖带图历史、替换图片、图文并发、取消恢复，
  与单独运行 token 相同，驻留后源读取为 0；16384 配置窗口、512 prefill、48-token 上限。
- W4A4 实验版：上述五类基础生成通过，上限 32 tokens；不据此承诺跨 CPU/GPU 放置一致性。
- 新应用实际加载 262144 配置窗口、8192 prefill、双路及 GPU 视觉的稳定模型，
  经本地 Harness 完成一次真实聊天，收到 `READY` 回复；截图在
  `work/upgrade-020beta/app-chat.png`。验收后退出模型与测试应用服务。

受管 Windows 生命周期专项已验证启动和显式 generation verify；完整回滚套件仍存在
失败的源配置迁移/回滚场景及临时日志被占用的清理错误。新原生应用走前台服务子进程，
此问题不作为已修复项。通用安装器的模拟后台进程测试亦未全部通过。

## 边界与回退

未做整机速度对测、全模型质量评估、满 262K prompt、MTP 开启后的本轮验收，
也未构建/运行 Linux、HIP、SYCL、Metal。上游发布的速度数字不能算作本机收益。
FP4-Local 的旧状态差异问题及 W4A4 的放置敏感性仍保留。

本机新增入口使用浏览器版应用界面；没有重新打包或覆盖官方 Tauri 安装程序。
原生配置的上下文/BF16/专家常驻方式从设置文件修改，界面暂不提供通用自动调优。
上游 `strata launch` 安装向导、签名组件更新和模型下载仍主要面向 GGUF，
不会把官方预编译引擎变成此原生后端。当前原生更新应从本仓库构建。

旧的稳定/实验程序保留。升级前 HEAD、工作区补丁、两份 BAT 和原始配置保存于
`G:/Strata/Strata-data/cleanup-records/upgrade-020beta-20261020/`。
回退时先退出新模型，再以备份配置启动对应旧程序；不要同时加载两份约 63 GiB 的专家 arena。
