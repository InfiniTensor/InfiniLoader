# InfiniLoader 使用指南

[返回首页](../README.zh-CN.md)

本指南使用自定义发布目录，首页快速开始使用默认目录。如果已按首页启动 `qwen`，
请先结束读取方并执行 `iloader stop qwen --remove-view`，再按本指南从头操作。

[English](usage.md)

InfiniLoader 把模型文件保存在后台内存缓存中，让推理进程重启后能复用这些文件，
减少从原始存储重复读取权重的耗时。它发布一个兼容普通文件读取的模型目录，
由框架原有的文件加载器读取；模型计算和请求调度仍由推理框架负责。

先准备一次缓存，再从下面两种方式中选择：

- 在引擎命令前加上 `iloader run 缓存名 --`，模型路径写成 `'{model}'`。
  InfiniLoader 自动填入缓存路径，并保护运行中的读取方。
- 直接把缓存发布的目录传给引擎。此时推理命令中不需要调用 `iloader`，也不用
  `'{model}'` 占位符。

下面统一使用来源目录 `/models/Qwen3-8B`、缓存名 `qwen`、发布目录
`$HOME/model-views/qwen`。请将来源改成已下载的本地模型目录；若修改缓存名或发布目录，
后续命令也应同步修改，`'{model}'` 则始终原样保留。
各框架命令是供选择的不同用法，不要把所有服务同时启动。

## 安装

要求 Linux 和 Python 3.10+。在本仓库目录中，将 InfiniLoader 安装到需要调用
`iloader` 的各个环境：

```bash
python3 -m pip install .
iloader doctor
iloader selftest
```

推理框架及其依赖需要另外安装。开发时可在本仓库执行 `source env.sh`，直接使用
源码并启用 Bash 补全。缓存进程与读取方须使用同一用户，处于同一 PID 命名空间，
并能访问缓存进程的 `/proc` 文件。运行目录必须位于本机，支持文件锁、Unix socket
和硬链接。容器配置等条件见[运维说明](operations.md)。

## 1. 先准备内存缓存

示例使用 Bash。下面的命令不会启动 GPU 或推理服务：

```bash
iloader inspect /models/Qwen3-8B
iloader start /models/Qwen3-8B --name qwen --view "$HOME/model-views/qwen"
iloader status qwen
iloader view qwen
```

`inspect` 查看所选文件大小和可用内存。`start` 将文件复制到内存，启动后台缓存
进程，等待缓存目录就绪后返回。`status` 查询状态，`view` 输出发布目录的绝对路径。
不填写 `--view` 时，InfiniLoader 会使用默认位置 `BASE/views/qwen`。这里的 `BASE`
表示运行根目录，通常是 `~/.cache/infiniloader`；它只是说明用的记号，不是需要你
定义的 shell 变量。自定义方法见第 5 节。

| 示例中的值 | 谁来填写 | 代表什么 |
| --- | --- | --- |
| `/models/Qwen3-8B` | 你填写 | 原始本地模型目录，包含权重、配置和 tokenizer 等文件。需要先下载，不能直接写在线模型 ID。 |
| `qwen` | 你通过 `--name qwen` 起名 | 本地缓存标识，后续供 `run`、`status`、`view`、`stop` 使用，与模型目录名和推理 API 的服务模型名无关。 |
| `$HOME/model-views/qwen` | 你通过 `--view` 指定 | 使用缓存时传给引擎的目录；`$HOME` 是普通 shell 变量，表示当前用户的主目录。 |

推理进程退出后，缓存进程仍然保留，下一次可直接复用；默认不开启空闲回收。
用同一名称再次启动未变化的来源和文件选择，会复用现有缓存。请使用不可变的模型
版本；来源内容或缓存进程设置变化时，应换一个名称，或显式停止后重建。

### `'{model}'` 到底是什么？

它是一个固定占位符，意思是：**“把这个缓存的目录路径填到这里。”**
你直接照抄，保留花括号；只有 `iloader run` 会识别并替换它：

```bash
iloader run qwen -- cat '{model}/config.json'
```

假设你的主目录是 `/home/alice`，使用上面创建的缓存后，实际启动的命令就是：

```text
cat /home/alice/model-views/qwen/config.json
```

所以，`qwen` 决定**使用哪个缓存**，`'{model}'` 决定**把缓存路径填到命令的哪里**。
它不是原始模型名，也不是 shell 变量。没有 `iloader run` 时，要直接填写真实路径，
推理引擎本身不会替换 `'{model}'`。

## 2. 选择启动方式

| | 通过 `iloader run` 启动 | 直接启动推理引擎 |
| --- | --- | --- |
| 是否需要先准备缓存 | 需要，先执行 `start`。 | 需要；也可以由运维或启动脚本提前执行 `start`。 |
| 模型参数写什么 | 字面量 `'{model}'`。 | 真实发布目录，例如 `"$HOME/model-views/qwen"`。 |
| 引擎的其他参数放哪里 | 保持在引擎命令后面，原样保留。 | 按原来的引擎用法填写。 |
| 如何保护读取方 | 前台子进程运行期间自动持有租约。 | 没有自动租约；保持空闲回收关闭，不要停止或回收仍有读取方的缓存。 |
| 推理时是否调用 iloader | 是。 | 否，只有准备缓存时需要。 |

包装启动时，命令顺序是：

```text
iloader run [ILOADER_OPTIONS] CACHE_NAME -- ENGINE_COMMAND [ENGINE_ARGUMENTS]
```

`run` 自身的选项（例如 `--base-dir`、`--socket`）放在**缓存名前面**。
`--` 后是实际启动的引擎命令：引擎选项、位置参数和带引号的值保留在原来的位置。
`run` 替换参数中的 `{model}` 后，将参数传给子进程，不会重新解释框架参数，
也不会通过 shell 执行整条命令。

`run` 还会给子进程设置 `ILOADER_MODEL_PATH` 环境变量。它是提供给子进程使用的
路径信息，手动设置这个变量并不会让推理引擎自动切换模型路径。

**能否保持原始磁盘路径，只加一个引擎参数就启用缓存？**
当前不能。仓库没有给下面三个引擎提供原生启用开关，也没有通过环境变量自动重定向
原始路径的功能。直接读取 `/models/Qwen3-8B`，读取的仍是来源文件。
应使用 `run`，或将引擎模型参数改为发布目录。兼容此文件接口的加载器无需修改源码。

## 3. 三个推理框架的完整示例

先按第 1 节准备 `qwen`，并选择相应框架和硬件支持的模型。每个框架都给出
**原来的命令**、**通过 run 使用缓存**、**直接传入缓存路径**三种形式，其他服务参数
保持一致。这些是接入示例，不代表所有版本的兼容矩阵；当前 CI 验证缓存服务和
参考示例，并未覆盖三个引擎的所有版本。

### InfiniLM

先按 [InfiniLM 安装说明](https://github.com/InfiniTensor/InfiniLM)配置环境。
在其仓库目录中运行，将下面 `cd` 的路径换成你的 InfiniLM 仓库实际路径。
原来直接读取磁盘模型的命令是：

```bash
cd /absolute/path/to/InfiniLM
python3 python/infinilm/server/inference_server.py \
  --model /models/Qwen3-8B --device nvidia --tp 1 \
  --host 127.0.0.1 --port 8000
```

通过 `iloader run` 使用缓存：前面加包装命令，仅将模型路径替换为占位符。

```bash
iloader run qwen -- python3 python/infinilm/server/inference_server.py \
  --model '{model}' --device nvidia --tp 1 \
  --host 127.0.0.1 --port 8000
```

不经过 `run`，直接让 InfiniLM 读取已准备的缓存目录：

```bash
python3 python/infinilm/server/inference_server.py \
  --model "$HOME/model-views/qwen" --device nvidia --tp 1 \
  --host 127.0.0.1 --port 8000
```

`--model` 是模型目录；`--device nvidia` 选择 InfiniLM 的 NVIDIA 平台，其他平台
按安装环境调整；`--tp` 是张量并行度；`--host`、`--port` 是服务监听设置。
这些都属于 InfiniLM，不是 InfiniLoader 参数，其他框架参数也照常保留在引擎命令中。

### vLLM

先[安装 vLLM](https://docs.vllm.ai/en/latest/serving/online_serving/)。
假设你原来的命令是：

```bash
vllm serve /models/Qwen3-8B \
  --tensor-parallel-size 1 --host 127.0.0.1 --port 8000
```

通过 `iloader run` 使用缓存，其他参数原样保留：

```bash
iloader run qwen -- vllm serve '{model}' \
  --tensor-parallel-size 1 --host 127.0.0.1 --port 8000
```

不在推理命令中调用 `iloader`，则直接传入缓存目录：

```bash
vllm serve "$HOME/model-views/qwen" \
  --tensor-parallel-size 1 --host 127.0.0.1 --port 8000
```

`serve` 后的位置参数是模型目录。张量并行度、监听地址、端口和其他
[vLLM 参数](https://docs.vllm.ai/en/latest/cli/serve/)继续由 vLLM 处理。
如果调用方依赖显式设置的服务模型名，也应保留该设置；缓存名不会替你配置推理 API
的模型名称。

### SGLang

先[安装 SGLang](https://docs.sglang.io/docs/basic_usage/send_request)。
原来的命令是：

```bash
python3 -m sglang.launch_server --model-path /models/Qwen3-8B \
  --tp 1 --host 127.0.0.1 --port 30000
```

通过 `iloader run` 使用缓存：

```bash
iloader run qwen -- python3 -m sglang.launch_server --model-path '{model}' \
  --tp 1 --host 127.0.0.1 --port 30000
```

或者直接传入缓存目录：

```bash
python3 -m sglang.launch_server --model-path "$HOME/model-views/qwen" \
  --tp 1 --host 127.0.0.1 --port 30000
```

`--model-path` 是模型目录；`--tp`、`--host`、`--port` 以及其他
[服务参数](https://docs.sglang.io/docs/advanced_features/server_arguments)原样传给 SGLang。

### 复用缓存、已有脚本和停止顺序

如果创建缓存时没有指定 `--view`，应查询实际路径，不必猜测默认目录：

```bash
VIEW=$(iloader view qwen)
vllm serve "$VIEW" --tensor-parallel-size 1 --port 8000
```

这里的 `VIEW` 才是 shell 变量，值是一条真实路径。已有脚本可将它填入 InfiniLM
的 `--model`、vLLM 的模型位置参数或 SGLang 的 `--model-path`；直接启动时没有
占位符替换。

先退出推理进程，缓存仍可供下一次启动使用。所有读取方结束后，如果需要释放缓存
内存，再执行：

```bash
iloader stop qwen --remove-view
```

这会停止缓存进程并移除发布目录，不会删除原始模型或本地磁盘缓存版本。
视图依赖缓存进程存活，只在对应主机及 PID 命名空间内有效；开启张量并行不会自动
把缓存分发到其他机器，多节点需要在各节点准备可访问的缓存。

## 4. 本地磁盘缓存与内存预算


如果来源是网络盘或其他慢存储，可以先创建本地磁盘缓存版本。磁盘缓存与长驻
内存缓存是两种不同的对象：

```bash
LOCAL=$(iloader cache /models/Qwen3-8B --name disk-qwen --cache-root /node-local/model-cache)
iloader cache-status /models/Qwen3-8B --name disk-qwen --cache-root /node-local/model-cache
iloader cache-list --cache-root /node-local/model-cache
iloader verify "$LOCAL"
iloader run --cache-path "$LOCAL" -- cat '{model}/config.json'
```

这里的 `cache --name disk-qwen` 仅为**磁盘**缓存版本命名，不会启动名为
`disk-qwen` 的内存缓存进程。此时使用 `run --cache-path "$LOCAL"`，也可以把示例中
的 `cat` 读取程序换成上面任一框架的命令。

`cache` 和 `cache-status` 支持 `--name`（或 `--cache-name`）、`--cache-root`、
`--base-dir`、`--pattern`、`--exclude`、`--json`。`cache` 还支持 `--refresh`
重建、`--cache-workers`（默认 4）、整数字节数的 `--chunk-size`（默认 67108864）
以及关闭进度输出的 `--quiet`。磁盘缓存根目录须位于来源目录之外。
仅使用磁盘缓存不会额外把整个模型预载到 memfd，文件页缓存驻留由操作系统管理。

如需再从本地磁盘版本加载到内存，可以使用 `start --local-cache`。这是准备 `qwen`
的另一种方式；如果第 1 节的缓存进程仍在运行，应先结束所有读取方，执行
`iloader stop qwen --remove-view`，再运行下面的命令。不能通过重复启动已有缓存进程
来修改加载设置或内存上限。

```bash
iloader start /models/Qwen3-8B --name qwen --view "$HOME/model-views/qwen" \
  --local-cache --cache-root /node-local/model-cache \
  --cache-name disk-qwen --max-memory-bytes 32GiB --reserve-bytes 4GiB \
  --max-resident-bytes 64GiB
iloader gc --resident --max-bytes 32GiB          # 仅预览
iloader gc --resident --max-bytes 32GiB --apply  # 在读取方退出后执行
```

这些上限只是示例，需按模型和框架实际需求调整。内存缓存包含所选文件字节数和
加载缓冲区；框架自身分配，以及读取方持有的映射或写时复制内存需要另外计算。
密封的 memfd 页面不可修改，但没有通过 mlock 锁定，内存压力和 swap 仍会影响延迟。

## 5. CLI 参数参考

下面的 `MODEL_DIR` 表示本地来源目录，`NAME` 表示缓存标识。名称必须以 ASCII
字母或数字开头，只能包含 ASCII 字母、数字、`.`、`_`、`-`，最多 96 个字符，
不能包含 `..`，在 `--base-dir` / `ILOADER_BASE_DIR` 选择的注册目录内唯一。

### 命令形式与选项

```text
iloader start MODEL_DIR --name NAME [START_OPTIONS]
iloader run [--base-dir PATH] [--socket PATH] NAME -- COMMAND [ARGUMENTS...]
iloader run --cache-path LOCAL_GENERATION -- COMMAND [ARGUMENTS...]
iloader status|view|manifest|stop NAME [OPTIONS]
```

`start` 必须填写 `--name`；`run` 使用**位置参数**填写缓存名，InfiniLoader 自身
的选项必须放在**缓存名前面**，`--` 后面填写实际执行的命令和框架参数。
例如，`start` 的 `--workers` 属于 InfiniLoader，而 `vllm serve` 后面的
`--tensor-parallel-size` 属于 vLLM。

| 命令 | 用途及常用选项 |
| --- | --- |
| `inspect MODEL_DIR` | 预览所选文件大小和可用内存；支持 `--pattern`、`--exclude`、`--json`。 |
| `start MODEL_DIR --name NAME` | 预加载文件并启动长驻缓存进程，选项见下表。 |
| `run NAME -- COMMAND ...` | 使用内存缓存启动前台读取程序；`--base-dir`、`--socket` 放在 `NAME` 前。仅用磁盘缓存时，以 `--cache-path` 代替名称。 |
| `status NAME` | 查看状态、缓存字节数及租约；`--json` 输出结构化结果。 |
| `view NAME` / `manifest NAME` | 输出缓存视图路径 / JSON 文件清单。 |
| `list` | 列出内存缓存；支持 `--base-dir`、`--timeout`（默认 1 秒）、`--json`。 |
| `stop NAME` | 停止缓存进程；`--remove-view` 同时移除发布目录。`--all` 选择使用默认注册路径的内存缓存，`--force` 可强制停止有活跃租约的缓存。 |
| `cache MODEL_DIR` / `cache-status MODEL_DIR` / `cache-list` | 创建、检查或列出本地**磁盘**缓存版本，见第 4 节。 |
| `verify LOCAL_GENERATION` | 使用 SHA-256 校验磁盘缓存版本；`--quick` 仅检查元数据；支持 `--json`。 |
| `gc` | 预览磁盘缓存回收；`--resident` 改为内存缓存，`--max-bytes SIZE` 指定目标容量，`--ttl SECONDS` 选择空闲缓存，`--apply` 才实际回收。 |
| `doctor` / `selftest` | 检查环境 / 验证 memfd 文件接口；`doctor` 支持 `--base-dir`、`--json`。 |
| `bench PATH` | 顺序文件读取测试；支持 `--repeat`（默认 3）、`--pattern`、`--exclude`、`--chunk-size`、`--json`。不会清空系统页缓存，也不是框架启动性能测试。 |
| `completion bash` | 输出 Bash 补全脚本。 |

`status`、`view`、`manifest`、`stop` 也支持 `--name NAME`、`--socket PATH`、
`--base-dir PATH`、`--timeout SECONDS`（默认 5 秒）。`gc` 支持 `--base-dir`、
磁盘缓存的 `--cache-root` 以及 `--json`；没有 `--apply` 时只预览，有租约的缓存
会被跳过。`--resident` 不能与 `--cache-root` 同时使用。

### start 参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--name NAME` | 必填 | 内存缓存标识。 |
| `--view PATH` | `BASE/views/NAME` | 发布缓存目录的位置，必须与来源目录分离。 |
| `--base-dir PATH` | 见下文 | 运行元数据、视图、日志和默认磁盘缓存的根目录。 |
| `--socket PATH` | `BASE/run/NAME.sock` | 自定义 Unix socket；后续命令（包括 `run`）需传入相同路径。 |
| `--pid-file PATH`、`--log-file PATH` | `BASE/run/NAME.pid`、`BASE/logs/NAME.log` | 自定义缓存进程 PID 文件和日志路径。 |
| `--pattern GLOB` | 全部文件，再应用排除规则 | 按相对路径选择文件，可重复；glob 应加引号，避免被 shell 提前展开。 |
| `--exclude GLOB` | 内置排除规则 | 按相对路径排除文件，可重复；一旦提供，就会**替换**默认排除列表。 |
| `--workers N` | `1` | 并行预加载内存的工作线程数，与 GPU 数量或张量并行度无关。 |
| `--chunk-size BYTES` | `67108864`（64 MiB） | 每个工作线程的复制缓冲区，只接受整数字节数，不能写 `64MiB`。 |
| `--max-memory-fraction FRACTION` | `0.90` | 可用内存准入检查使用的比例。 |
| `--max-memory-bytes SIZE` | 不设置 | 单个缓存的预加载容量上限，包含工作线程缓冲区。 |
| `--reserve-bytes SIZE` | `256MiB` | 准入检查保留的余量，需按框架需求调整；不代表实际预留物理内存。 |
| `--max-resident-bytes SIZE` | 不设置 | 为 socket 的父目录（通常是 `BASE/run`）设置持久化的共享内存缓存配额。 |
| `--idle-ttl SECONDS` | 不启用 | 空闲多久后回收缓存；通过 `run` 租约保护活跃读取方。 |
| `--force` | 关闭 | 跳过可用内存检查，仍受显式单缓存上限和共享配额限制。 |
| `--wait-timeout SECONDS` | `600` | 等待缓存进程就绪或注册的时限，不包含本地磁盘缓存准备阶段。 |
| `--no-wait` | 关闭 | 注册完成后即返回，此时视图可能尚未就绪。 |
| `--local-cache` | 关闭 | 先创建或复用本地磁盘缓存版本，再载入内存。 |
| `--cache-root PATH` | `BASE/model-cache` | 启用 `--local-cache` 时的磁盘缓存根目录，须位于来源目录之外。 |
| `--cache-name NAME` | 内存缓存的 `NAME` | `--local-cache` 使用的磁盘缓存标签。 |
| `--refresh-cache` | 关闭 | 在 `--local-cache` 准备阶段重建磁盘缓存版本。 |
| `--cache-workers N` | `4` | `--local-cache` 阶段的并行磁盘复制线程数。 |

`SIZE` 支持整数字节数和 `512MiB`、`32GiB` 等二进制单位。
`BASE` 优先使用 `ILOADER_BASE_DIR`，否则使用 `$XDG_CACHE_HOME/infiniloader`
或 `~/.cache/infiniloader`，必要时回退到临时目录。要让不同命令和推理环境使用同一
注册目录，可统一设置 `export ILOADER_BASE_DIR=/node-local/path/infiniloader`，
或在支持的命令中传入相同的 `--base-dir`。Unix socket 路径编码后不能超过 107 字节。

建议保留默认 socket 和 PID 路径。自定义 socket 的父目录会形成独立的内存配额组，
仅使用相同 `--base-dir` 不会合并这些配额。注册目录范围的发现和清理依赖默认路径：
自定义 PID 文件可能使进程无法被列出，`stop --all` 也不会自动沿用自定义 socket。
此类进程应显式使用 `iloader stop --socket PATH NAME` 停止。

文件选择应保留配置和 tokenizer 等文件。仅设置 `--pattern '*.safetensors'`
可能导致框架找不到必需的元数据。各命令完整选项可通过 `iloader COMMAND --help` 查看。

## 文件缓存的实现

核心只依赖 Python 标准库，将所选文件复制到密封的 `memfd`，再通过指向
`/proc/<daemon-pid>/fd/<fd>` 的链接发布文件目录。框架的内存分配和 GPU 传输仍由
框架负责；文件缓存不代表重启时的每个阶段都会加速。

## 更多文档

- [运维、校验、回收及容器配置](operations.md)
- [参与开发与运行测试](../CONTRIBUTING.md)
- [可选的张量分片、TensorStore 与专家示例](distributed.md)
- [围绕模型缓存的开发范围](roadmap.zh-CN.md)
- [历史测量与维护归档](archive/README.md)

TensorStore、显式 TP/PP 布局和 GPU 专家池属于可选实验 API，依赖额外的 PyTorch
环境。参考 MLP/MoE 示例验证数值行为，不提供生产模型适配器、量化内核或推理引擎。
核心 `start/run` 文件缓存流程不需要这些 API。

[MIT 许可证](../LICENSE)。
