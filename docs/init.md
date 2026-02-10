# 项目入口分析

## 概述

Nano-vLLM 的项目入口设计非常简洁，主要由三个文件组成：
- `nanovllm/__init__.py`：包入口，导出公共 API
- `nanovllm/llm.py`：LLM 类定义（用户 API）
- `nanovllm/engine/llm_engine.py`：LLMEngine 类（核心引擎）

## 文件结构

### nanovllm/__init__.py

```python
from nanovllm.llm import LLM
from nanovllm.sampling_params import SamplingParams
```

这是包的入口文件，对外暴露两个核心类：
- `LLM`：主要的推理接口
- `SamplingParams`：采样参数配置

### nanovllm/llm.py

```python
from nanovllm.engine.llm_engine import LLMEngine

class LLM(LLMEngine):
    pass
```

`LLM` 类是 `LLMEngine` 的直接子类，没有添加任何额外的方法。这种设计的原因：
1. 保持用户 API 的简洁性（用户只需使用 `LLM`）
2. 为未来扩展预留空间（可在 LLM 层添加高级功能）
3. 分离用户接口和实现细节

### nanovllm/engine/llm_engine.py

这是核心引擎文件，定义了 `LLMEngine` 类，包含了所有推理相关的逻辑。

## LLM 初始化流程

### 构造函数

```python
def __init__(self, model, **kwargs)
```

**参数**：
- `model`：模型路径（HuggingFace 格式）
- `**kwargs`：可选配置参数，包括：
  - `max_num_batched_tokens`：最大批处理 tokens 数
  - `max_num_seqs`：最大序列数
  - `max_model_len`：最大模型长度
  - `gpu_memory_utilization`：GPU 内存利用率（默认 0.9）
  - `tensor_parallel_size`：张量并行大小（默认 1）
  - `enforce_eager`：是否强制使用 eager 模式


`huggingface格式`：这是最常用的格式，由 用户名（或组织名）/ 项目名 组成。格式： owner/model_name


**初始化步骤**：

1. **创建 Config 对象**
   ```python
   config = Config(model, **config_kwargs)
   ```
   - 过滤出 Config 类定义的字段
   - 验证配置参数

2. **初始化多进程 ModelRunner（张量并行）**
   ```python
   for i in range(1, config.tensor_parallel_size):
       event = ctx.Event()
       process = ctx.Process(target=ModelRunner, args=(config, i, event))
       process.start()
   ```
   - 使用 `spawn` 上下文创建子进程
   - 为每个并行 worker 创建进程和事件对象
   - 从 rank=1 开始创建（rank=0 由主进程处理）

3. **初始化主进程 ModelRunner**
   ```python
   self.model_runner = ModelRunner(config, 0, self.events)
   ```
   - rank=0 的 ModelRunner 在主进程中运行
   - 负责协调所有 worker 进程

4. **加载 Tokenizer**
   ```python
   self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
   config.eos = self.tokenizer.eos_token_id
   ```

5. **初始化 Scheduler**
   ```python
   self.scheduler = Scheduler(config)
   ```

6. **注册退出处理**
   ```python
   atexit.register(self.exit)
   ```
   - 确保进程正确清理

## 对外 API 接口

### 1. generate() - 主要生成接口

```python
def generate(
    self,
    prompts: list[str] | list[list[int]],
    sampling_params: SamplingParams | list[SamplingParams],
    use_tqdm: bool = True,
) -> list[str]
```

**功能**：批量生成文本

**参数**：
- `prompts`：提示列表，可以是字符串列表或 token id 列表
- `sampling_params`：采样参数，可以是单个对象或列表（与 prompts 一一对应）
- `use_tqdm`：是否显示进度条

**返回值**：
```python
[{"text": "生成的文本", "token_ids": [token_id_list]}, ...]
```

**执行流程**：
1. 初始化进度条（如果 `use_tqdm=True`）
2. 标准化采样参数（如果是单个参数，复制到所有提示）
3. 添加所有请求到调度器
4. 进入主循环：
   - 调用 `step()` 执行单步推理
   - 更新进度条和性能指标（prefill/decode throughput）
   - 收集完成的输出
5. 返回解码后的结果

### 2. add_request() - 添加单个请求

```python
def add_request(self, prompt: str | list[int], sampling_params: SamplingParams)
```

**功能**：添加单个推理请求到调度器

**参数**：
- `prompt`：提示字符串或 token id 列表
- `sampling_params`：采样参数

**逻辑**：
1. 如果是字符串，使用 tokenizer 编码
2. 创建 Sequence 对象
3. 添加到调度器

### 3. step() - 单步推理

```python
def step(self) -> tuple[list, int]
```

**功能**：执行单步推理（prefill 或 decode）

**返回值**：
- `outputs`：完成的序列列表 `[(seq_id, token_ids), ...]`
- `num_tokens`：处理的 token 数（正数为 prefill，负数为 decode）

**执行流程**：
1. 调度器选择要执行的序列：`seqs, is_prefill = self.scheduler.schedule()`
2. 模型推理：`token_ids = self.model_runner.call("run", seqs, is_prefill)`
3. 后处理：`self.scheduler.postprocess(seqs, token_ids)`
4. 收集完成的序列

### 4. is_finished() - 检查完成状态

```python
def is_finished(self) -> bool
```

**功能**：检查所有请求是否已完成

### 5. exit() - 清理资源

```python
def exit(self)
```

**功能**：清理进程和资源

**逻辑**：
1. 通知所有 ModelRunner 退出
2. 删除主进程 ModelRunner
3. 等待所有子进程结束

## LLM 类与 LLMEngine 的关系

### 继承关系

```
LLMEngine (核心引擎实现)
    ↑
    |
LLM (用户接口)
```

### 设计模式

这种设计采用了 **Facade（外观模式）** 和 **Inheritance（继承模式）** 的组合：

1. **LLM 作为外观层**：
   - 提供简洁的用户 API
   - 隐藏复杂的内部实现

2. **继承关系**：
   - LLM 直接继承 LLMEngine
   - 没有添加额外方法，保持透明
   - 未来可在 LLM 层添加高级功能

### 优势

1. **简洁性**：用户只需了解 LLM 类
2. **灵活性**：LLMEngine 可以独立测试和使用
3. **可扩展性**：未来可在 LLM 层添加：
   - Chat 模板支持
   - 流式输出
   - 高级采样策略
   - 多轮对话管理

### 与 vLLM 的对比

vLLM 的设计类似：
- `vllm.LLM` 是用户接口
- `vllm.engine.llm_engine.LLMEngine` 是核心实现
- 但 vLLM 的 LLM 不是直接继承，而是组合关系

Nano-vLLM 选择了更简单的继承关系，代码更紧凑（约 1200 行 vs vLLM 数万行）。

## 使用示例

### 基本用法

```python
from nanovllm import LLM, SamplingParams

# 初始化 LLM
llm = LLM(
    "/path/to/model",
    tensor_parallel_size=1,
    enforce_eager=True
)

# 配置采样参数
sampling_params = SamplingParams(
    temperature=0.6,
    max_tokens=256
)

# 生成文本
prompts = ["Hello, Nano-vLLM.", "Tell me a joke."]
outputs = llm.generate(prompts, sampling_params)

# 输出结果
for output in outputs:
    print(output["text"])
```

### 高级用法

```python
# 批量输入不同的采样参数
prompts = ["Write a poem.", "Write a story."]
sampling_params = [
    SamplingParams(temperature=0.8, max_tokens=100),  # 更随机
    SamplingParams(temperature=0.2, max_tokens=500),  # 更确定
]
outputs = llm.generate(prompts, sampling_params)

# 使用 token id 输入
prompt_tokens = [15496, 11, 428, 318, 262, 4494]  # "Hello, world"
outputs = llm.generate([prompt_tokens], sampling_params)

# 不使用进度条
outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
```

## 总结

Nano-vLLM 的项目入口设计体现了以下原则：

1. **简洁性**：通过 LLM 简化用户 API
2. **模块化**：清晰分离用户接口和核心实现
3. **可扩展性**：预留扩展空间
4. **高效性**：直接继承，无额外开销

这种设计使得 Nano-vLLM 既保持了 vLLM 的 API 兼容性，又大幅简化了代码实现。