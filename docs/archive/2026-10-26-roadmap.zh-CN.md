# InfiniLoader 后续能力路线

评估日期：2026-10-26。以下是基于当前实现和官方上游接口的开发建议，尚未实现，也不是性能承诺。

## 建议定位

InfiniLoader 适合继续发展成供推理引擎复用的权重存储与驻留层：管理内容身份、不可变代际、进程间共享、布局和设备副本生命周期。计算图、算子、KV cache 和请求调度仍由推理引擎负责。下一轮最有价值的交付是一个真实模型适配器和可信的端到端性能证据。

当前已经具备 sealed memfd、租约、磁盘代际、cgroup 内存预算、显式 TP/PP 切片、共享 CPU TensorStore 和同步 GPU ExpertPool。四机测试证明了小型 MLP 的 TP/PP 数值路径及参考 MoE 换入换出；它们没有验证生产 LLM、量化引擎兼容性、GPU 动态拓扑切换或分布式专家并行。

## 优先级与验收条件

| 优先级 | 能力 | 当前缺口与具体交付 | 验收依据 |
| --- | --- | --- | --- |
| P0 | 真实模型与引擎适配 | 固定一个模型族和引擎版本，从 PyTorch/Transformers 接入开始，再扩展 vLLM/InfiniLM；显式处理 GQA、packed QKV、bias、tied weights、dtype 和非连续 slice | 与原生加载比较 logits/生成结果；验证 TP/PP；重启和重复加载；实际模型的冷启动、热启动与内存结果 |
| P0 | 端到端性能与物理内存观测 | 将读盘、hash、映射、参数安装、转换、H2D、引擎初始化和首 token 分开计时；增加共享 PSS、cgroup、GPU allocated/reserved 与 NUMA 观测 | 同模型、同设备、同软件版本下比较原生 mmap、热 page cache、InfiniLoader；报告重复样本及 p50/p95，不用读文件带宽代替启动收益 |
| P1 | 批量 catalog、导出和复用 | 当前 export_rank 在导出前后分别 catalog 全量源；TensorStore 也会校验全量内容。设计一次 snapshot/catalog、多 rank 有界导出和代际级校验复用 | 记录总读字节与 hash 字节；rank 数增加时避免重复全量扫描；源变化、进程退出、损坏和重试测试仍通过 |
| P1 | 引擎原生格式与 KTransformers 适配 | 按内容 ID、格式、转换参数和内核版本缓存派生产物；提供原生 tensor/格式适配，明确 C++ 内核是否复制、持有还是借用存储 | 一个真实 MoE 端到端运行；精度检查、格式失效和重启复用；同时报告 CPU canonical、转换副本、GPU 权重与总显存 |
| P1 | 有界异步预取与冷热策略 | 当前 promote 同步，hot set 由调用方指定，参考执行按专家串行。先增加有界 staging、异步句柄、取消/失败状态，再对接路由统计和原生 CPU 内核 | 相同质量和显存限制下测 TTFT、每 token 延迟、命中率、H2D 字节、等待和抖动；证明计算/传输实际重叠，不只统计 promotion 次数 |
| P1 | 节点级服务与稳定客户端协议 | 提供版本化 acquire/release/describe 接口、多模型幂等 warmup、可查询操作状态、指标和能力探测；评估 FD 传递以减少对跨容器 /proc 路径的依赖 | 同节点多进程共享；不同 PID namespace 的可支持配置测试；daemon 崩溃、客户端存活映射及 COW 对租约和配额的影响明确 |
| P2 | 分布式分片迁移与重分布 | 现有 sharded store 绑定原 layout/rank；完整副本可本地重切片，但不提供旧分片到新分片的在线迁移。先做可信、可核验、有界传输，再考虑 RDMA | TP4→TP2/PP2→PP4 的真实模型迁移；校验所需 shard 完整后才发布；中断可恢复，重复请求不重复启动，不足内存时拒绝；GPU 拓扑与通信单独实测 |

## 先解决真实加载链路中的复制

共享 CPU Tensor 并不意味着引擎会直接复用该 storage。常规 load_state_dict 可能复制到已分配参数，量化、packing 和 pinning 也会产生新存储。适配器应先测量这些步骤，明确参数及 storage 的所有权，再选择 meta 初始化和参数绑定方式。PyTorch 的官方加载说明展示了 mmap、meta 与 assign=True 的组合；这可作为适配切入点，并不代表所有模型和版本都能直接套用。[PyTorch 加载指南](https://docs.pytorch.org/tutorials/recipes/recipes/module_load_state_dict_tips.html)

vLLM 已有按 worker 加载自身 TP/PP state 的原生 loader，因此接入要实现其期望的参数语义和存储格式。InfiniLoader 的显式 layout 并不是该格式的直接替代品。[vLLM ShardedStateLoader](https://docs.vllm.ai/en/latest/api/vllm/model_executor/model_loader/sharded_state_loader/)

## 与 KTransformers 的合作边界

KT-Kernel 已有原生格式、CPU 内核、NUMA 配置、GPU 专家选择和动态放置相关接口，也公开了从 tensor 加载权重的入口。应优先适配和测量这些入口，让 InfiniLoader 提供可复用的权重及派生产物管理；不要将另一个参考 LRU/MoE 调度器视为主要产品收益。接入后仍需验证原生内核的布局、复制和异步使用契约，不能由 PyTorch 视图的共享性质推断原生内核零复制。[KT-Kernel 官方说明](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/README.md)

## 分布式模式如何演进

固定布局且 RAM 紧张时优先分片；调试和频繁拓扑变更时完整 CPU 副本更方便。后续可以依据内存预算、预计切换频率与迁移成本选择模式，记录选择理由，但不应在缺少带宽和工作负载测量时宣称自动最优。

拓扑重分布需要参数语义、数据移动、协调和故障恢复共同工作。PyTorch Distributed Checkpoint 可作为状态分片与恢复接口的参考，不能把它的状态恢复能力等同于 InfiniLoader 已支持在线推理拓扑迁移。[PyTorch Distributed Checkpoint](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html)

实验中的共享目录可见性问题还说明，节点上的权威操作状态和可查询结果比仅依赖目录 marker 更可靠。可继续接入既有算力调度平台的资源分配和租约，把权重准备作为作业阶段；InfiniLoader 保持权重层职责，不复制建设整套集群调度系统。

## 推荐实施顺序

1. **真实模型接入与基线。** 一个稠密模型、一个引擎、固定版本，完成结果正确性和原生/缓存加载对比，同时补齐 storage 复制与内存观测，并给出可复现安装、版本兼容和迁移说明。
2. **减少重复准备与原生 MoE 接入。** 批量 catalog/export、派生格式缓存，再接一个 KT-Kernel 原生路径，量化收益与内存成本。
3. **按瓶颈推进异步和集群能力。** 有性能证据后做预取、NUMA、热点策略；有可靠节点服务后再做在线重分布，最后评估 RDMA。

现阶段不建议同时承诺所有量化格式、所有推理引擎、多租户服务、训练状态缓存和通用 GPU 共享。它们具有不同的所有权与一致性要求，应由具体适配需求驱动。
