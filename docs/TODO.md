# Nano-vLLM 源码阅读计划

## 阅读目标

通过系统化的阅读顺序，深入理解 Nano-vLLM 的设计理念和实现细节，掌握 LLM 推理引擎的核心技术。

---

## 第一阶段：基础架构与数据结构（约30分钟）

**目标**：理解项目的基本结构和核心数据结构

### 1. 项目入口
- **文件**：`nanovllm/__init__.py` 和 `nanovllm/llm.py`
- **重点**：
  - 如何初始化 LLM 类
  - 对外提供的 API 接口
  - LLM 类与 LLMEngine 的关系

### 2. 配置系统
- **文件**：`nanovllm/config.py`
- **重点**：
  - Config 类的参数定义
  - 配置验证和初始化逻辑
  - 关键参数的含义（max_num_batched_tokens、max_model_len、tensor_parallel_size 等）

### 3. 采样参数
- **文件**：`nanovllm/sampling_params.py`
- **重点**：
  - SamplingParams 类的参数
  - 采样约束条件（temperature 必须 > 1e-10）
  - max_tokens 和 ignore_eos 的作用

---

## 第二阶段：核心引擎组件（约45分钟）

**目标**：理解推理引擎的核心组件及其职责

### 4. 序列管理
- **文件**：`nanovllm/engine/sequence.py`
- **重点**：
  - Sequence 类的设计
  - 序列状态机（WAITING/RUNNING/FINISHED）
  - block_table 和 KV cache 映射机制
  - `__getstate__` 和 `__setstate__` 的序列化优化

### 5. KV Cache 管理
- **文件**：`nanovllm/engine/block_manager.py`
- **重点**：
  - Block 类和引用计数机制
  - Prefix Caching 的哈希匹配实现（xxhash）
  - 块的分配和释放逻辑
  - `allocate`、`deallocate`、`can_allocate` 方法

### 6. 调度器
- **文件**：`nanovllm/engine/scheduler.py`
- **重点**：
  - 调度策略（prefill vs decode）
  - 抢占机制（preemption）
  - 资源约束管理（max_num_batched_tokens、max_num_seqs）
  - `schedule` 方法的逻辑流程

---

## 第三阶段：推理执行流程（约45分钟）

**目标**：理解推理请求的完整执行流程

### 7. LLM 引擎
- **文件**：`nanovllm/engine/llm_engine.py`
- **重点**：
  - `__init__` 方法：多进程 ModelRunner 初始化
  - `generate` 方法：主循环逻辑和输出处理
  - `step` 方法：单步推理流程（调度→推理→后处理）
  - 进程管理和事件同步

### 8. 模型运行器
- **文件**：`nanovllm/engine/model_runner.py`
- **重点**：
  - 张量并行初始化（NCCL、SharedMemory）
  - KV Cache 内存分配策略
  - `run` 方法：推理执行的入口
  - `prepare_prefill` 和 `prepare_decode`：数据准备逻辑
  - 进程间通信机制（write_shm、read_shm）

---

## 第四阶段：底层算子与优化（约60分钟）

**目标**：理解底层的算子实现和性能优化技术

### 9. 注意力机制
- **文件**：`nanovllm/layers/attention.py`
- **重点**：
  - Flash Attention 2 集成（flash_attn_varlen_func、flash_attn_with_kvcache）
  - KV Cache 存储的 Triton kernel（store_kvcache）
  - Paged Attention 实现（通过 block_table 和 slot_mapping）
  - prefill 和 decode 两种模式的区别

### 10. 采样器
- **文件**：`nanovllm/layers/sampler.py`
- **重点**：
  - Temperature scaling 的实现
  - Gumbel-Max 采样算法
  - torch.compile 优化应用

### 11. 并行线性层
- **文件**：`nanovllm/layers/linear.py`
- **重点**：
  - QKVParallelLinear：Q、K、V 并行计算
  - MergedColumnParallelLinear：门控和上投影合并
  - RowParallelLinear：行并行线性层
  - 张量并行通信（all_reduce）

### 12. 其他组件
- **文件**：`nanovllm/layers/`
- **重点**：
  - `activation.py`：SiluAndMul 激活函数
  - `layernorm.py`：RMSNorm 实现
  - `rotary_embedding.py`：旋转位置编码
  - `embed_head.py`：VocabParallelEmbedding 和 ParallelLMHead

---

## 第五阶段：模型实现（约45分钟）

**目标**：理解具体的模型架构实现

### 13. Qwen3 模型
- **文件**：`nanovllm/models/qwen3.py`
- **重点**：
  - Qwen3Attention：注意力层实现
    - QKV 并行投影
    - 旋转位置编码应用
    - 分组查询注意力（GQA）
  - Qwen3MLP：前馈网络实现
  - Qwen3DecoderLayer：解码器层的完整流程
  - Qwen3ForCausalLM：完整模型的组装
  - 模块打包策略（packed_modules_mapping）

---

## 第六阶段：关键特性深入（约60分钟）

**目标**：深入理解关键优化技术的实现细节

### 14. CUDA Graph 优化
- **文件**：`nanovllm/engine/model_runner.py`（capture_cudagraph 方法）
- **重点**：
  - 如何捕获不同 batch size 的 CUDA Graph
  - graph_bs 列表的设计（1, 2, 4, 8, 16, 32, ..., 512）
  - 如何在推理时重用 CUDA Graph
  - 何时使用 eager 模式而非 CUDA Graph

### 15. Prefix Caching 完整流程
- **追踪路径**：从哈希计算到块复用
- **重点**：
  - 如何计算块的哈希值
  - 如何匹配相同内容的块
  - 引用计数的管理
  - 缓存命中时如何节省计算

### 16. 张量并行通信
- **文件**：`nanovllm/engine/model_runner.py` 和 `nanovllm/utils/context.py`
- **重点**：
  - 主进程（rank=0）的协调角色
  - Worker 进程的命令循环
  - SharedMemory 和 Event 的使用
  - NCCL 通信的集成

### 17. 工具函数
- **文件**：`nanovllm/utils/`
- **重点**：
  - `loader.py`：模型加载逻辑
  - `context.py`：全局上下文管理（Context 类）

---

## 第七阶段：实践验证（约30分钟）

**目标**：通过实际运行验证理解

### 18. 运行示例
- **文件**：`example.py`
- **重点**：
  - 如何使用 LLM API
  - 如何使用 chat template
  - 输出格式

### 19. 运行基准测试
- **文件**：`bench.py`
- **重点**：
  - 性能测试方法
  - 如何测量吞吐量
  - prefill 和 decode 的性能指标

---

## 阅读建议

### 每个文件阅读时关注的问题

1. **类/函数的职责是什么？**
2. **它与其他组件如何交互？**
3. **有哪些关键的优化技术？**

### 推荐的阅读工具

- 使用 IDE 的跳转功能（如 VS Code 的 Cmd+Click）
- 使用 ripgrep (`rg`) 搜索关键函数调用
- 使用断点调试（推荐使用 `pdb` 或 IDE 调试器）

### 关键概念清单

阅读完所有文件后，你应该能够回答以下问题：

- [ ] LLM 推理的完整数据流是什么？
- [ ] KV Cache 如何管理？什么是 Paged KV Cache？
- [ ] Prefix Caching 如何工作？如何提升性能？
- [ ] 调度器如何决定哪些序列执行？什么是抢占？
- [ ] 张量并行如何实现？进程间如何通信？
- [ ] Flash Attention 2 与传统注意力有什么区别？
- [ ] CUDA Graph 如何优化性能？什么情况下使用？
- [ ] prefill 和 decode 阶段有什么区别？

### 时间安排

- **总预计时间**：4-5 小时
- **建议节奏**：每次阅读 1-2 个阶段，避免一次性阅读过多

---

## 扩展阅读

### 相关论文

- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)
- [PagedAttention: Efficient KV Cache Management for LLM Serving](https://arxiv.org/abs/2309.06180)
- [vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180)

### 参考项目

- [vLLM](https://github.com/vllm-project/vllm)：Nano-vLLM 的灵感来源
- [FlashAttention](https://github.com/Dao-AILab/flash-attention)：Flash Attention 的实现

---

## 进度跟踪

| 阶段 | 状态 | 完成日期 | 备注 |
|------|------|----------|------|
| 第一阶段 | ☐ | | |
| 第二阶段 | ☐ | | |
| 第三阶段 | ☐ | | |
| 第四阶段 | ☐ | | |
| 第五阶段 | ☐ | | |
| 第六阶段 | ☐ | | |
| 第七阶段 | ☐ | | |

---

## 问题记录

在阅读过程中，如果你遇到难以理解的概念或代码，可以记录在这里：

- [问题 1]：
  - 文件：`xxx.py`
  - 行号：`xxx`
  - 描述：
  - 解决方法/备注：

- [问题 2]：
  - 文件：`xxx.py`
  - 行号：`xxx`
  - 描述：
  - 解决方法/备注：

---

## 笔记区域

用于记录阅读过程中的关键发现和思考：

### 关键发现 1
（记录时间：____）

### 关键发现 2
（记录时间：____）

---

*祝阅读愉快！如有疑问，随时提问。*