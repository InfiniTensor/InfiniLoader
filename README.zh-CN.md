# InfiniLoader

[English](README.md)

InfiniLoader 把模型文件保存在长期运行的缓存进程中，让推理引擎反复重启时复用权重，减少对原始存储的重复读取。它是 Linux 用户态缓存，负责权重文件及其生命周期；模型计算和请求调度由推理引擎承担。

核心仅依赖 Python 标准库。选中的文件被复制到封存的 memfd，并通过指向 `/proc/<daemon-pid>/fd/<fd>` 的目录链接供现有文件加载器读取。

## 安装与环境检查

需要 Linux 和 Python 3.10+。建议安装在引擎所在环境，与消费者使用相同用户和 PID namespace：

```bash
python3 -m pip install .
iloader doctor
iloader selftest
```

开发时可以使用 `source env.sh`，直接加载源码并启用 Bash 补全。运行目录应位于节点本地，并支持文件锁、Unix socket 和硬链接；`doctor` 会检查这些能力。

## 加载一次，重复使用

```bash
MODEL=/absolute/path/to/model
iloader start "$MODEL" --name dev
iloader status dev

# 示例：已安装 vLLM 时使用；仍需验证实际引擎版本。
iloader run dev -- vllm serve '{model}' --tensor-parallel-size 1

# 引擎退出后缓存继续保留，可重复运行。
iloader run dev -- vllm serve '{model}' --tensor-parallel-size 1
iloader stop dev --remove-view
```

`run` 替换参数中的字面量 `{model}`，并设置 `ILOADER_MODEL_PATH`。消费者前台运行期间有租约保护。InfiniLoader 选项放在缓存名之前；消费者应在其 worker 需要读取 view 时保持存活。`stop` 等待原 daemon 退出，超时则返回错误。

文件接口是集成边界。引擎格式、单文件路径解析、容器和加载器行为都可能影响兼容性，必须验证实际版本。目前 CI 覆盖缓存服务与参考示例，没有覆盖按版本维护的 vLLM/SGLang/InfiniLM 兼容矩阵。

## 磁盘缓存与内存限制

仅使用本地磁盘代际，不额外分配整份模型大小的 RAM：

```bash
LOCAL=$(iloader cache "$MODEL" --name dev)
iloader verify "$LOCAL"
iloader run --cache-path "$LOCAL" -- vllm serve '{model}'
```

需要从本地代际继续预加载到 RAM 时，使用 `start --local-cache`。磁盘缓存根目录必须位于源模型目录之外。

```bash
iloader start "$MODEL" --name dev --local-cache \
  --max-memory-bytes 32GiB --reserve-bytes 4GiB --max-resident-bytes 64GiB
iloader gc --resident --max-bytes 32GiB          # 仅预览
iloader gc --resident --max-bytes 32GiB --apply
```

上述容量只是示例，应按模型和引擎实际占用设置。缓存计费包括文件内容与加载缓冲，引擎分配和客户端仍持有的映射/COW 副本另计。memfd 已封存但未 mlock，内存压力和 swap 仍会影响延迟。

源模型应使用不可变版本。源变化后需更换缓存名或先停止再重建。直接使用 `iloader view dev` 路径的读者没有自动租约，应保持自动 idle eviction 关闭。

## 文档与可选接口

- [操作、校验、回收与容器](docs/operations.md)
- [贡献与测试](CONTRIBUTING.md)
- [可选分片、TensorStore 和专家示例](docs/distributed.md)
- [围绕缓存目标的开发范围](docs/roadmap.zh-CN.md)
- [历史测量与维护资料](docs/archive/README.md)

TensorStore、显式 TP/PP layout 和 GPU 专家池是实验接口，使用可选 PyTorch 依赖。参考 MLP/MoE 验证基本数值行为，不提供生产模型适配器、量化内核或 serving engine；核心 `start/run` 不需要这些接口。

完整 CLI 以 `iloader --help` 和 `iloader COMMAND --help` 为准。常用核心命令为 `start`、`run`、`status`、`list`、`stop`、`cache`、`verify`、`gc`、`doctor` 和 `bench`。`bench` 不清除 OS page cache，仅测文件读取，不能等同于引擎启动收益。

[MIT 许可](LICENSE)。
