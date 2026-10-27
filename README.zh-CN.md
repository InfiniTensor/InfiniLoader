# InfiniLoader

[English](README.md)

面向 Linux 的模型文件缓存。将模型文件载入后台内存缓存后，可以在推理进程多次
重启之间复用，通过普通文件接口读取；兼容的加载器无需修改框架源码。

## 安装

要求 Linux 和 Python 3.10+。在本仓库目录中安装到推理环境，推理框架需另外安装。

```bash
python3 -m pip install .
iloader doctor
```

## 快速开始

先从已有的本地模型目录创建缓存：

```bash
iloader start /models/Qwen3-8B --name qwen
```

将 `/models/Qwen3-8B` 换成你的模型目录。`qwen` 是你为缓存起的名字。
命令等待缓存就绪后返回；推理进程退出后，缓存仍然保留。

### 启动推理引擎

选择已安装且支持该模型的引擎。在原命令前加上 `iloader run qwen --`，
把模型路径替换为 **`'{model}'`**，其他引擎参数原样保留。
`'{model}'` 直接照抄，InfiniLoader 会自动将它填成缓存路径。

**InfiniLM**：在你的 [InfiniLM 仓库](https://github.com/InfiniTensor/InfiniLM)中运行：

```bash
cd /absolute/path/to/InfiniLM
iloader run qwen -- python3 python/infinilm/server/inference_server.py \
  --model '{model}' --device nvidia --tp 1 --host 127.0.0.1 --port 8000
```

将 `cd` 路径换成实际仓库路径，`--device` 按硬件调整。

**vLLM：**

```bash
iloader run qwen -- vllm serve '{model}' \
  --tensor-parallel-size 1 --host 127.0.0.1 --port 8000
```

**SGLang：**

```bash
iloader run qwen -- python3 -m sglang.launch_server --model-path '{model}' \
  --tp 1 --host 127.0.0.1 --port 30000
```

重复执行所选命令即可复用缓存。`run` 在前台子进程运行期间持有租约，防止缓存被
正常回收或停止；只要工作进程仍需读取文件，就应保持这个前台子进程存活。
请按实际环境验证框架版本兼容性，上述示例不代表完整兼容矩阵。

### 直接使用已有启动命令

也可以取得缓存目录，直接传给引擎，不经过 `iloader run`：

```bash
VIEW=$(iloader view qwen)
vllm serve "$VIEW" --tensor-parallel-size 1 --port 8000
```

InfiniLM 对应 `--model "$VIEW"`，SGLang 对应 `--model-path "$VIEW"`。
直接启动没有自动租约：保持空闲回收关闭，读取期间不要停止或回收缓存。
原始磁盘路径不会自动转向缓存，当前也没有引擎原生参数用于启用 InfiniLoader。

### 管理缓存

```bash
iloader status qwen
iloader list
# 所有读取方退出后，释放内存并移除缓存视图。
iloader stop qwen --remove-view
```

停止缓存不会删除原始模型文件。缓存进程和读取方须使用同一用户及 PID 命名空间，
容器配置见[运维说明](docs/operations.md)。

## 分布式使用（实验性）

多节点场景需要在每个节点运行缓存。实验性的 `iloader shard` 命令支持按
**显式 TP/PP 布局**导出各 rank 的权重分片；张量切分规则、文件分发和各节点缓存
启动需要使用者安排，尚未提供自动跨机部署。导出的格式需要相应模型适配器，
不能直接替换上面框架示例中的原生模型目录。详见[分布式用法](docs/distributed.md#deployment-boundary)。

## 性能

缓存减少对来源存储的重复读取，但需要预载时间和内存。系统页缓存已经热时，额外
缓存的加速收益可能很小。框架初始化和 CPU 到 GPU 的传输仍有开销，应结合缓存
准备成本，测量自己环境中的完整工作流。

## 文档

- [使用指南与 CLI 参数](docs/usage.zh-CN.md)：完整命令对照、`start` 参数、内存预算和磁盘缓存。
- [运维说明](docs/operations.md)：校验、回收及容器配置。
- [参与开发](CONTRIBUTING.md)：开发环境与测试。
- [实验 API](docs/distributed.md)：可选的张量分片、TensorStore 和专家示例。

命令帮助：`iloader --help` 或 `iloader COMMAND --help`。

[MIT 许可证](LICENSE)。
