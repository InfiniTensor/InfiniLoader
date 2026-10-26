# 提交线重整与验证

日期：2026-10-26（Asia/Shanghai）。

将 master `5014a6a` 之后的 19 个工作提交重整为 11 个主题提交。原远程备份
`codex/all-changes-backup-20261020` 及原功能分支保留；不改写其历史，不合并 master。
新分支为 `improve-weight-residency`。新提交均使用一句英文 subject，Author 和
Committer 均为 `whjthu <haojie0429@gmail.com>`。

## 新提交顺序与独立验证

测试专家对每个精确 hash 单独执行 git archive，在 Linux 原生临时目录运行完整
`python3 -B -m pytest -q`，未依赖变化中的工作树。设置
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`、`PYTHONDONTWRITEBYTECODE=1`。环境为 WSL Ubuntu、
Python 3.14.4、PyTorch 2.14.1+cpu、NumPy 2.5.4、safetensors 0.8.0、pytest 9.1.1。

| 顺序 | 提交 | 一句话 subject | 独立完整测试原始摘要 |
| --- | --- | --- | --- |
| 1 | `ea57e3c` | Protect sealed weight snapshots and daemon runtime ownership | 40 passed in 7.09s |
| 2 | `63e7ea4` | Bound IPC requests and cancel failed distributed reads | 46 passed in 11.71s |
| 3 | `97335f6` | Publish verified disk cache generations with lease-aware collection | 62 passed in 12.15s |
| 4 | `8cca2e6` | Enforce container memory limits and shared resident budgets | 69 passed in 12.73s |
| 5 | `e2c2065` | Add leased commands and cache lifecycle diagnostics | 77 passed in 25.08s |
| 6 | `7454e9c` | Document cache operations and add Linux regression CI | 77 passed in 23.76s |
| 7 | `48615de` | Add explicit tensor sharding with atomic streaming export | 102 passed in 27.24s |
| 8 | `43d7a75` | Share sealed CPU tensors with live topology slicing | 108 passed in 31.83s |
| 9 | `70c0979` | Validate distributed TP and PP execution after coordinated preload | 116 passed in 78.24s (0:01:18) |
| 10 | `895c5ef` | Manage GPU expert residency with CPU fallback and safe eviction | 119 passed, 2 skipped in 80.49s (0:01:20) |
| 11 | 本文件所在提交 | Document validated history and development priorities | 文档与来源映射；最终 hash 的审核结果随交付报告给出 |

逐提交元数据、原始 pytest 日志 SHA-256 及完整映射见
[history-reorganization.json](history-reorganization.json)。原始日志保存在本次任务的
`InfiniLoader-commit-review-20261026` 证据目录；该本地目录不是公开下载链接。

生命周期专家审核了 1–5；缓存/内存专家审核了 1、3、4、7、8、10；CLI/验证专家审核
提交依赖、元数据、逐提交完整测试及 6、9 的 CI/验证流程。每个提交都须完成对应独立
审核后才推送。最终文档提交还需独立终审，此处不预先记为已批准。

## 为什么这样重整

- 把运行文件所有权、PID/state 原子更新和 inode 清理放在启动主题中；把 prepare 后与
  daemon load 后的身份复核一并提前，避免中间提交加载新内容却发布旧身份。
- 把单 chunk 内存约束纳入快照拷贝，把完整 inventory/schema 和真实容量回收纳入磁盘代际。
- 分片功能首次出现即具备 NFS 发布与中断/EIO 保护；TP/PP 示例首次出现即带启动屏障。
- TensorStore 先于专家池；CI 的 hybrid 示例步骤随实际示例一起引入。

## 实现等价与历史证据

第 10 提交 `895c5ef` 的 `src/`、`examples/`、`.github/`、
`deploy/` 和 `pyproject.toml` Git 对象与原终版 `88d6bac` 完全相同。最终提交仅调整文档。
测试唯一差异是故障注入用例的启动等待窗由 3 秒改为 20 秒；源位于 Windows 挂载目录时
首次完整回归在 worker 报告损坏前已耗尽 3 秒，因此出现与预期错误类型不同的 barrier
超时。错误类型、损坏检测和全部子进程回收断言均保留；修正随首次引入 TP/PP 测试提交。
可独立复核：

```bash
git diff --exit-code 88d6bac improve-weight-residency -- src examples .github deploy pyproject.toml
git diff 88d6bac improve-weight-residency -- tests
```

原四机 CPU/GPU 测量保留其真实代码版本（例如 `123c6e2`、`758f8c8`），不改写为新 hash。
源码相同说明实现被保留，不代表本次重新执行了 GPU、RCCL 或四机实验。两个真实设备
测试在本地 CPU 环境跳过，原设备验证见 [distributed-validation.md](distributed-validation.md)。
当前分支的远端 CI 应查看该分支自己的 Actions；原报告中的旧 Actions 链接仍只证明旧提交。

## 原提交归属

| 原提交 | 新主题顺序 | 归属说明 |
| --- | --- | --- |
| `2f7204b` | 1 | Startup serialization and runtime ownership |
| `c67ba9e` | 1 | Sealed immutable snapshots |
| `deaa940` | 2 | Bounded IPC and failed-rank cancellation |
| `d726100` | 1, 3 | Identity rechecks moved to startup; verified disk generations |
| `2f54480` | 4 | Cgroup and shared resident budgets |
| `fb2f057` | 5 | Leased execution and diagnostics |
| `8fab6cd` | 1, 2, 3, 4, 5 | Boundary, stop, GC and budget corrections folded into owners |
| `a17a6e9` | 5 | Benchmark medians and cache uncertainty |
| `6a33d83` | 6 | Operations and core CI |
| `7e1b248` | 1, 2, 5 | Runtime ownership, cancellation diagnostics and lease-safe stop |
| `cc67790` | 1, 3 | Bounded copy buffers and complete disk generation verification |
| `7442271` | 5, 6 | Regression alignment and historical expert evidence |
| `d8fe312` | 7 | Strict explicit sharding |
| `4ab6d4a` | 8 | CPU TensorStore |
| `674f168` | 10 | GPU expert pool and hybrid reference |
| `717997e` | 9, 10 | TP/PP tests; hybrid CI smoke kept with its example |
| `123c6e2` | 7 | NFS publication fix included in initial shard feature |
| `758f8c8` | 9 | Startup barrier included in initial TP/PP harness |
| `88d6bac` | 11 | Distributed contracts, historical numerical evidence and README |

后续能力优先级与验收条件见 [roadmap.zh-CN.md](roadmap.zh-CN.md)。
