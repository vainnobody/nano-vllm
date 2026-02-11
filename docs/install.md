# 环境安装指南

## 前置要求

- Python 3.10-3.12（不支持 3.13+）
- CUDA 环境（用于 flash-attn 编译）
- Conda（推荐使用 Miniconda 或 Anaconda）

## 1. 配置 Conda 清华镜像源

```bash
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/pro
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
conda config --set show_channel_urls yes
```

## 2. 创建虚拟环境

```bash
conda create -n nano-vllm python=3.12 -y
conda activate nano-vllm
```

## 3. 安装依赖

### 3.1 安装主要依赖

```bash
pip install torch==2.4.0 triton==3.0.0 transformers==4.51.0 xxhash -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 3.2 安装 flash-attn

flash-attn 需要从源码编译，需要 CUDA 编译环境：

```bash
pip install flash-attn --no-build-isolation
```

如果编译失败，可以尝试使用预编译版本（需要根据 CUDA 版本选择）：

```bash
# CUDA 12.1
pip install flash-attn==2.6.3 --no-build-isolation
```

## 4. 安装项目

```bash
cd /path/to/nano-vllm
pip install -e .
```

## 5. 验证安装

```bash
python -c "import nanovllm; print('安装成功')"
```

## 常见问题

### flash-attn 安装失败

确保已安装 CUDA Toolkit 和 nvcc：

```bash
nvcc --version
```

如果未安装，使用 conda 安装：

```bash
conda install cuda-toolkit
```

### Python 版本不兼容

nano-vllm 需要 Python 3.10-3.12，不支持 3.13+。如果当前版本是 3.13，请使用 conda 创建 3.12 环境。