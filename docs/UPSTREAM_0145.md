# 上游 0.1.45 融合：2026-10-18

融合官方 `v0.1.45`（`477ac13c`），本地源码基线为 `be40a26c`。
这是 Windows / RTX 5090 / Ryzen 9950X3D / 96 GB RAM 的原生 safetensors 桌面更新。

## 保留的桌面路径

- 主模型直接读取原始 NVFP4 safetensors；全部专家保持锁定 RAM 副本和加载后读取封闭。
- 8192-token W4A4/CUTLASS auto prefill，BF16 KV Tensor Core attention，原 FP32 激活专家 decode。
- 原始 MTP 权重，单路及双路 batch MTP；物理核布局、独立伸缩 KV、设备槽复制与图片身份隔离。
- 262144 配置窗口，12 GiB 内存会话缓存、unlimited 槽位，端口 `127.0.0.1:8880`。
- 保留文件中已有的 temperature 1.0、top_p 0.95、top_k 20，以及单路/双路各自的 prefill overlap 设置。
- `STRATA_ROUTE_TAIL_SKIP=0`、`STRATA_COMMIT_BATCH=0`，未启用本地 `STRATA_SLOT_COMMIT_BATCH`。

单路和双路入口使用单独生成的 `build-native-engine/strata-classic-bf16-kv-v0145.exe`。
原有程序保留。桌面 BAT 的名称和调用链不变；构建脚本的默认输出同步到该文件。

## 融合边界

同步上游服务端参数、增量 prompt 编码、工具调用解析、监控、模型元数据及 CUDA/CPU 更新。
保留本地 BF16 激活投影、BF16 KV、W4A4 prefill、原生 MTP、连续/重叠 prefill 与会话分支保护。

上游重构了 QSA chunk/merge 调用及直接槽复制。BF16 KV 接入新的 chunk 调用，
后续 merge 沿用共享实现；原生槽复制改用上游统一的 checkpoint/KV copy 接口，
保留逐会话 M-RoPE。移除自动合并产生的重复函数和测试目标。
本地 in-place MMQ 与上游 pointer-table MMQ 使用互斥选择，避免同一组同时采用两套寻址。

上游的多卡/多用户和其他实验开关维持其选择方式；本次桌面配置没有额外开启它们。
官方 Qwen3.6 或其他 GPU 的速度数字不作为本机原生 Flash-Next 的收益。

## 验证

工具链：Windows，CUDA 13.0，MSVC 14.44，SM120。原生引擎及选定测试目标构建通过。

- 服务端 unittest：861 项，15 项跳过，无失败（293.177 s，mock engines）。
- 原生 reader：13 项，1 项跳过；通用 tokenizer：11 项，3 项跳过，均无失败。
- 原始 HF tokenizer：43 个对照及 30 个增量编码用例通过。
- 缓存、快照、CUTLASS 和 prefill 选择器：11 项 CTest 通过。
- BF16 attention：27 个 oracle/graph 用例通过；槽提交：672 个 graph replay 用例通过。
- 真实模型单路：MTP 关闭和两枚草稿各 17 个完成请求，包含中英文、代码、seeded sampling、
  输出上限、EOS、A/B/A、磁盘恢复、取消及续写。两组完成输出和主模型状态一致，
  驻留后专家/MTP 源读取均为 0。使用 262144 配置窗口、8192 prefill、BF16 KV，
  视觉编码器加载；请求较短，草稿门槛为 0，以覆盖验证/回滚。

- 真实模型双路：10 个场景通过，包括 greedy/seeded sampling、惩罚参数、不同草稿置信度、
  取消恢复、followup、短输出上限、EOS，以及 9511-token 长槽位增长/复用旁路接纳。
  并发输出与各自单独运行一致；16384 配置窗口、8192 prefill、BF16 KV、原生 batch MTP、
  自适应热专家晋升、视觉编码器、独立 KV 和设备槽复制均开启。
  驻留后的专家/MTP 源读取均为 0，私有引擎正常退出。

- 图片会话：22 个请求、6 组并发组合通过，覆盖 greedy/seeded 图片并发、带图历史续聊、
  图片与文本并发、替换图片、取消及恢复；并发输出与各自单独运行一致。
  本项使用 16384 配置窗口、512 prefill、48-token 输出上限；未运行 extended 图片长历史测试。
  驻留后专家/MTP 源读取均为 0，私有引擎正常退出。

原始日志和 JSON 位于忽略目录 `work/upgrade-0145/`。
所有验收程序已退出；本次没有留下后台模型服务。

没有做升级前后的整机速度对测、全面任务质量评估或满 262K prompt 测试。
HIP、SYCL、Metal 和 Linux 未在本机构建或运行；继承支持不等于已完成验证。

## 本地改动与回退

升级前的工作区补丁、两份桌面配置和 stash 标识保存在
`G:/Strata/Strata-data/cleanup-records/upgrade-0145-20261018/`。
原有未提交配置修改及删除的旧 GGUF launcher 保留；没有恢复已删除的模型。
