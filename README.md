# OpenPI-AUBO i5 Scripts

这个仓库只包含机器人侧脚本，负责：

- 在线推理执行
- 真机数据采集
- 观测构建
- AUBO 控制辅助逻辑
- 任务完成判断

默认目录形态：

```text
openpi/
├─ repo/        # OpenPI 主仓库
├─ OpenPi-Auboi5/
├─ aubo_sdk/
└─ captures/
```

## 仓库结构

```text
OpenPi-Auboi5/
├─ config       # 根目录单文件配置
├─ tools/
├─ utils/
├─ support/
├─ task/
├─ data/
├─ main.py
└─ collect_data.py
```

`utils/` 放环境、路径和通用工具。
`support/` 放机器人控制、策略加载、观测、TUI 和 observer 相关逻辑。

## 快速开始

1. 直接编辑根目录 `config`

至少确认这些字段：

- `OPENPI_RUNTIME_PYTHON`
- `OPENPI_SDK_ROOT`
- `OPENPI_ROBOT_IP`
- `OPENPI_CHECKPOINT_DIR`
- `OPENPI_TASK_OBSERVER_PYTHON`
- `OPENPI_TASK_OBSERVER_MODEL`
- 串口相关配置

2. 运行环境检查：

```bash
python3 tools/doctor.py
```

只检查主运行环境：

```bash
python3 tools/doctor.py --section runtime
```

只检查 observer 环境：

```bash
python3 tools/doctor.py --section observer
```

3. 启动在线推理：

```bash
python3 main.py
```

4. 启动数据采集：

```bash
python3 collect_data.py
```

## 配置加载规则

默认会按下面顺序加载配置：

1. `OPENPI_ENV_FILE` 指向的显式文件
2. 仓库根目录下的 `config`

现在不再使用 `config/` 目录存放 `.env` 模板。

## 环境依赖

这个仓库不是完整单仓环境，运行前仍需要准备外部依赖。

### 目录依赖

默认要求以下目录与当前仓库并列存在：

- `../repo`
- `../aubo_sdk`

### Python 环境

推荐两套 Python：

- OpenPI 主环境
  由 `OPENPI_RUNTIME_PYTHON` 指向
- Observer / Gemma 环境
  由 `OPENPI_TASK_OBSERVER_PYTHON` 指向

本地 PyTorch worker 也可以单独用一套环境，由 `OPENPI_PYTORCH_RUNTIME_PYTHON` 指向。

### 系统和设备依赖

运行前通常还需要：

- Jetson + JetPack / CUDA
- AUBO SDK
- Orbbec Python SDK
- 串口设备
- Gemma 模型目录

`tools/doctor.py` 会尽量提前暴露缺项，但不替代系统层安装。

## 主要入口

- `main.py`: 在线推理执行入口
- `collect_data.py`: 真机数据采集入口
- `support/load_policy.py`: 本地 / 远端策略统一加载
- `support/get_obs.py`: 观测构建
- `support/task_observer.py`: 任务完成判断

## PyTorch Local Backend

本地 PyTorch 推理当前通过独立 worker 进程运行：

- 主进程继续负责相机、机器人和主控制逻辑
- PyTorch policy worker 运行在 `OPENPI_PYTORCH_RUNTIME_PYTHON` 指向的环境
- worker 环境需要能加载 CUDA Torch 和转换后的 `model.safetensors`

常用变量：

- `OPENPI_POLICY_BACKEND=pytorch`
- `OPENPI_PYTORCH_CHECKPOINT_DIR=...`
- `OPENPI_PYTORCH_RUNTIME_PYTHON=/home/niic/openpi/miniforge3/envs/openpi-py310-torch/bin/python`
- `OPENPI_PYTORCH_DEVICE=cuda`
- `OPENPI_SAMPLE_NUM_STEPS=5`

## 兼容说明

- `support/kubeconfig.yaml` 按敏感文件处理，默认不纳入 git
