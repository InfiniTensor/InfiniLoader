# InfiniLoader

[English](README.md)

InfiniLoader 将模型文件保存在长驻内存缓存中，让推理进程重启后可以复用这些文件，
减少从原始存储重复读取权重的耗时。它是 Linux 用户态文件缓存；模型计算和请求调度
仍由推理框架负责。

核心仅依赖 Python 标准库，将选中的文件复制到密封的 `memfd`，再发布一个文件链接
指向 `/proc/<daemon-pid>/fd/<fd>` 的目录。框架将这个目录作为模型路径，继续使用原有
文件加载器。缓存进程独立于推理进程，可以在多次启动之间复用。

## 1. 安装

要求 Linux 和 Python 3.10+。在 InfiniLoader 仓库目录中，将它安装到需要使用
`iloader` 的各个推理环境：

```bash
python3 -m pip install .
iloader doctor
iloader selftest
```

开发时也可以在本仓库执行 `source env.sh`，直接使用源码并启用 Bash 补全。
推理框架需要另外安装。缓存进程与读取方须使用同一用户，处于同一 PID 命名空间，
并能访问缓存进程的 `/proc` 文件。运行目录必须位于本机，支持文件锁、Unix socket
和硬链接；`doctor` 会检查这些条件。容器配置见[运维说明](docs/operations.md)。

## 2. 先区分模型目录、缓存名和缓存路径

| 写法 | 含义 | 示例 |
| --- | --- | --- |
| `MODEL` / `MODEL_DIR` | 已经存在的**本地模型目录**，包含权重、配置和 tokenizer 等文件。需要先下载模型，不能直接填写在线模型 ID。 | `/mnt/models/Qwen3-8B` |
| `--name NAME` | 为内存缓存指定的本地标识，供 `run`、`status`、`stop` 使用。它与模型目录名、推理 API 的服务模型名没有绑定关系。 | `--name dev` |
| `VIEW` | 缓存进程发布的目录，其中的文件指向内存中的缓存内容。`start` 和 `view` 输出其绝对路径。 | `~/.cache/infiniloader/views/dev` |
| `'{model}'` | **由 `iloader run` 识别的字面占位符**。保留花括号，`run` 将命令参数中的它替换为缓存目录。它不是 shell 变量，也不是模型 ID。 | `--model '{model}'` |

缓存名必须以 ASCII 字母或数字开头，只能包含 ASCII 字母、数字、`.`、`_`、`-`，
最多 96 个字符，且不能包含 `..`。名称在 `--base-dir` / `ILOADER_BASE_DIR`
选择的运行注册目录内唯一。

**使用缓存时，传给框架的模型路径需要指向缓存视图。** 当前没有对原始磁盘目录做
透明读取拦截：直接把 `/mnt/models/Qwen3-8B` 传给框架，读取的仍是原始文件。
请通过 `run` 使用 `'{model}'`，或显式传入 `VIEW` 路径。兼容此文件接口的加载器
无需修改框架代码。

## 3. 先学会独立使用 iloader

下面的 Bash 命令不依赖推理框架，先完成缓存准备、检查和一次文件读取：

```bash
MODEL=/absolute/path/to/model
iloader inspect "$MODEL"
iloader start "$MODEL" --name dev
iloader status dev
iloader list
iloader view dev

# 小型读取程序：打印替换后的路径，并读取缓存中的配置文件。
iloader run dev -- python3 -c \
  'from pathlib import Path; import sys; p = Path(sys.argv[1]); print(p); print((p / "config.json").read_text()[:200])' \
  '{model}'
```

这个小程序要求模型目录中有 `config.json`。默认情况下，`start` 等待缓存视图就绪
后返回，标准输出只有视图路径，进度写入标准错误；因此也可以使用
`VIEW=$(iloader start "$MODEL" --name dev)` 保存返回路径。

读取程序退出后，缓存进程仍然保留。重复执行 `iloader run dev -- ...` 即可复用。
对同一名称再次启动未变化的来源目录和文件选择，会复用现有缓存。请使用不可变的
模型版本；来源不同或内容变化时，需要换一个缓存名，或显式停止后重建。
需要调整已有缓存进程的加载设置时，也应先停止再启动。

`run` 在前台子进程运行期间持有租约，防止缓存被正常回收或停止。只要工作进程
还需要文件，就应保持这个前台子进程存活。它还会给子进程设置 `ILOADER_MODEL_PATH`
环境变量，直接执行参数而不经过 shell，并返回子进程的退出状态。外层 shell 变量
会在 `run` 启动前展开，因此示例使用字面量 `'{model}'`。

### 命令形式与参数位置

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
| `cache MODEL_DIR` / `cache-status MODEL_DIR` / `cache-list` | 创建、检查或列出本地**磁盘**缓存版本，见第 5 节。 |
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

## 4. InfiniLM、vLLM、SGLang 如何加载缓存

先按第 3 节准备好 `dev`，并在各自环境中安装框架，选择该框架支持的模型。
以下示例请**分别运行**；兼容的加载器可以复用同一份文件缓存。这些是接入示例，
不代表完整的框架版本兼容矩阵：当前 CI 验证缓存服务和参考示例，未覆盖所有框架版本。

### InfiniLM

先按 [InfiniLM 安装及服务说明](https://github.com/InfiniTensor/InfiniLM)配置环境。
在 **InfiniLM 仓库目录**中运行，并确保此环境能调用 `iloader`：

```bash
cd /absolute/path/to/InfiniLM
iloader run dev -- python3 python/infinilm/server/inference_server.py \
  --model '{model}' --device nvidia --tp 1
```

`--model` 接收内存缓存目录；`--device nvidia` 选择 InfiniLM 的 NVIDIA 平台，
使用其他平台时按实际安装环境调整；`--tp 1` 是框架的张量并行度。
其他 InfiniLM 服务参数照常添加。

### vLLM

先[安装 vLLM](https://docs.vllm.ai/en/latest/serving/online_serving/)，然后运行：

```bash
iloader run dev -- vllm serve '{model}' --tensor-parallel-size 1
```

`serve` 后的位置参数是模型目录，`--tensor-parallel-size 1` 是 vLLM 的张量并行度。
其他 vLLM 参数照常写在框架命令后面。

### SGLang

先[安装 SGLang](https://docs.sglang.io/docs/basic_usage/send_request)，然后运行：

```bash
iloader run dev -- python3 -m sglang.launch_server \
  --model-path '{model}' --tp 1
```

`--model-path` 接收缓存目录；`--tp 1` 是 SGLang 的张量并行度。
其他设置见其[服务参数说明](https://docs.sglang.io/docs/advanced_features/server_arguments)。

### 在已有脚本中显式使用缓存路径

```bash
VIEW=$(iloader view dev)
# 示例：在已有 vLLM 命令中只替换模型路径。
vllm serve "$VIEW" --tensor-parallel-size 1
```

InfiniLM 对应 `--model "$VIEW"`，SGLang 对应 `--model-path "$VIEW"`。
直接启动**不会自动持有租约**：应关闭空闲回收，并在读取方运行期间避免停止或回收
缓存。需要自动管理生命周期时，优先使用 `iloader run`。
视图依赖缓存进程存活，只在对应主机及 PID 命名空间内有效；开启张量并行并不会
自动把缓存分发到其他机器，多节点需要在每个节点准备可访问的缓存。

所有读取方退出后，可以释放内存并移除视图：

```bash
iloader stop dev --remove-view
```

这不会删除原始模型目录，也不会删除已创建的本地磁盘缓存版本。

## 5. 可选的本地磁盘缓存与内存预算

如果来源是网络盘或其他慢存储，可以先创建本地磁盘缓存版本。磁盘缓存与长驻
内存缓存是两种不同的对象：

```bash
MODEL=/absolute/path/to/model
LOCAL=$(iloader cache "$MODEL" --name local-dev --cache-root /node-local/model-cache)
iloader cache-status "$MODEL" --name local-dev --cache-root /node-local/model-cache
iloader cache-list --cache-root /node-local/model-cache
iloader verify "$LOCAL"
iloader run --cache-path "$LOCAL" -- python3 -c \
  'from pathlib import Path; import sys; print((Path(sys.argv[1]) / "config.json").read_text()[:200])' \
  '{model}'
```

这里的 `cache --name local-dev` 仅为**磁盘**缓存版本命名，不会启动名为
`local-dev` 的内存缓存进程。此时使用 `run --cache-path "$LOCAL"`，也可以把示例中
的 Python 读取程序换成上面任一框架的命令。

`cache` 和 `cache-status` 支持 `--name`（或 `--cache-name`）、`--cache-root`、
`--base-dir`、`--pattern`、`--exclude`、`--json`。`cache` 还支持 `--refresh`
重建、`--cache-workers`（默认 4）、整数字节数的 `--chunk-size`（默认 67108864）
以及关闭进度输出的 `--quiet`。磁盘缓存根目录须位于来源目录之外。
仅使用磁盘缓存不会额外把整个模型预载到 memfd，文件页缓存驻留由操作系统管理。

如需再从本地磁盘版本加载到内存，可以使用 `start --local-cache`：

```bash
iloader start "$MODEL" --name dev --local-cache --cache-root /node-local/model-cache \
  --cache-name local-dev --max-memory-bytes 32GiB --reserve-bytes 4GiB \
  --max-resident-bytes 64GiB
iloader gc --resident --max-bytes 32GiB          # 仅预览
iloader gc --resident --max-bytes 32GiB --apply  # 在读取方退出后执行
```

这些上限只是示例，需按模型和框架实际需求调整。内存缓存包含所选文件字节数和
加载缓冲区；框架自身分配，以及读取方持有的映射或写时复制内存需要另外计算。
密封的 memfd 页面不可修改，但没有通过 mlock 锁定，内存压力和 swap 仍会影响延迟。

## 更多文档

- [运维、校验、回收及容器配置](docs/operations.md)
- [参与开发与运行测试](CONTRIBUTING.md)
- [可选的张量分片、TensorStore 与专家示例](docs/distributed.md)
- [围绕模型缓存的开发范围](docs/roadmap.zh-CN.md)
- [历史测量与维护归档](docs/archive/README.md)

TensorStore、显式 TP/PP 布局和 GPU 专家池属于可选实验 API，依赖额外的 PyTorch
环境。参考 MLP/MoE 示例验证数值行为，不提供生产模型适配器、量化内核或推理引擎。
核心 `start/run` 文件缓存流程不需要这些 API。

[MIT 许可证](LICENSE)。
