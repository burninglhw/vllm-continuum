# 原版 vLLM / Continuum 的受限 GPU 对比

这套工具比较**未经修改的 vLLM 0.10.2 Python 源码**与本仓库中按公开论文重建的 Continuum。双方共用已安装的 CUDA 扩展、构建时安装的 FlashAttention 依赖、模型及其他 Python 依赖；不是两套分别重编译的发行版。论文策略一致性的范围与未公开细节见 `continuum-conformance-v6.zh.md`，不能把实验成功当成作者未公开实现逐项一致的证明。

## 工具职责

| 文件 | 用途 |
|---|---|
| `tools/continuum/run_checkout.py` | 选择原版/修改版 Python 源码，仅设置子进程导入路径，不重装环境 |
| `tools/continuum/checkout_overlay/sitecustomize.py` | 显式复用 CUDA/FlashAttention 构建依赖，禁止调度器或引擎 Python 回退到另一仓库 |
| `tools/continuum/check_checkout.py` | 不加载模型权重，核对真正导入的调度器、引擎、Llama 及构建依赖的路径和哈希 |
| `tools/continuum/guard_gpu.py` | 记录显存和进程采样；越界时只终止自己新建的进程组 |
| `tools/continuum/profile_prefill.py` | 同硬件、模型、dtype、eager 和分块配置下标定完整上下文 prefill 成本 |
| `tools/continuum/verify_gpu.py` | 完整多轮真实前向回放；记录 JCT、prefill、缓存、抢占和策略估计阶段 |
| `tools/continuum/compare_checkouts.py` | 固定成对运行计划，seed42 先原版、seed43 先 Continuum；任一次失败则停止后续实验 |

## 数据和指标的口径

- DeepSeek 生成的是已有 agent **trace**；GPU 实际执行模型为本地 Llama-3.1-8B-Instruct BF16。不是在 H200 上重新运行 DeepSeek API，也不比较解题正确率。
- 每步先完成真实模型前向，再强制输出已有 trace token；两边 token 和工具耗时相同。不会把未来工具耗时或程序总轮数注入调度策略。
- 每个程序保留所有轮次、上下文、原始工具等待。上下文越界直接失败，不静默截断。工具按上一轮实际完成时间加原始 gap 返回。
- seed 只控制初始到达的指数间隔；不抽取另一批数据。两边每次都从空统计历史启动。
- JCT 从该程序预定到达计至最后一轮及其工具等待结束；包括排队、推理、工具时间，不包括模型加载和合成请求预热。
- 实际 prefill token 包括执行过的重复计算；不是简单用总输入减缓存命中来代替。request cache hit 和 admission cache hit 不混用；原版没有的 recorder 字段为 null。
- 同时记录无未释放 pin / KV 引用的检查结果。EOS 和所有输出 token 必须与记录完全相同。

## 共享 GPU 的保护边界

当前驱动设置本次全部 GPU 进程采样显存上限 **28 GiB**、空闲底线 **100 GiB**、PyTorch allocator 上限 **26 GiB**。KV 容量另行明确设定，主实验为 **4 GiB**。

`gpu_memory_utilization=0.2` **不是显存硬上限**：显式指定 `kv_cache_memory_bytes` 时，vLLM 会跳过其自动 KV 预算。PyTorch allocator 限额也不包含所有 CUDA/NCCL 等分配，因此还有外部进程显存采样保护。

保护器大约每 2 秒采样一次：出现新外部 GPU 进程、已有外部进程显存增加超过 512 MiB、可观测到的外部 SM/显存活动、自己的显存超预算或空闲低于底线时，停止本次进程组。它从不停止其他人的进程，不改 MPS/MIG、不设置系统级限速。

这不是硬件隔离，也不能保证两次采样之间没有瞬时越界。`pmon` 中的 `-` 记录为 null，不解释成已证明为 0；外部计算不可归因时，无法保证检测到所有干扰。正式论文测量应争取独占设备。

原版 vLLM 0.10.2 的内部设备映射只接受数字 GPU 序号。保护器先按 UUID 查当前索引，再设置 PCI 排序和数字 CUDA_VISIBLE_DEVICES；模型加载前再次用 PyTorch UUID 验证选卡。

## 2026-09-21 的输入与启动

服务器主目录：

```text
/export/home/ext.luohaowen1/continuum/reproduction/continuum-paper/gpu-compare-v6-20260921
```

原版源码归档 SHA-256：`57608f44cf61f5d80fb182c98e06e524cb2925bb528258a7b247c8e43a52d13e`。

数据集：`/export/home/ext.luohaowen1/continuum/reproduction/continuum-paper/datasets/deepseek-clean10-16k-20260921/replay.pkl.gz`。它包含 10 个完整程序、163 轮、153 次非最终工具返回；上下文最大 15043 tokens。原始 100 个程序中，16K 范围内且无恢复响应、符合解析器和 EOS 约定的全部合格程序只有这 10 个。本小集不是原论文完整工作负载，也不能代表被排除的长上下文/恢复型程序。

先用 `profile_prefill.py` 实测 1000、2000、4000、8000、16000、16383 上下文长度；每个长度一次预热、五次计入样本。prefill 标定关闭 APC，分块 2048 tokens；正式回放开启 APC。

成对运行的完整参数与实际命令保存在 `kv4/plan.json`，每次运行保存 `config.json`、`device.json`、`turns.jsonl`、`jobs.json`、`summary.json` 和 `policy-events.json`。旁边的 `*-monitor.jsonl` 记录从启动到退出的显存保护证据。原版不需要 TTL profile，但也记录相同 profile 文件哈希，便于对照。

无需为换版本重新安装环境。重跑时应使用新的输出目录，并先确认 GPU UUID/空闲资源；不能直接覆盖旧结果。临时原版源码和编译缓存可以重新从保存的源码归档构建；不要把 `/tmp` 路径当作长期唯一副本。

## 本次结果与勘误

四次正式回放均完成且输出逐token核对通过。两次种子均值：原版平均JCT=43.499秒、Continuum=47.747秒（+9.76%）；任务吞吐0.14689/s与0.12933/s（−11.95%）；实际prefill tokens均值714677.5与858152（+20.08%）。这不是论文加速结果复现。

两个Continuum运行分别产生153个TTL决策，三级估计分支均覆盖，但全部TTL=0、pin=0。实际prefill成本较低，独立日志复核表明所有正TTL的效用不如0；不能为了“跑出收益”改变公式或工具耗时。

实际测试脚本曾用按程序分组日志末尾的值作为`eta_last`，该值不是时间顺序最后的eta，不应引用。性能分析没有使用它。保存`tested-code.tar.gz`后，未来运行的汇总已改为`eta_final_empirical`；原始结果和性能指标保持不变。源码快照、原始结果、失败尝试和显存监测均保存在上述实验主目录。当前统计窗口/eta等未公开细节的限制仍按一致性核对文档披露。
