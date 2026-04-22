# OpenPI-AUBO i5 Scripts

这个仓库只包含 Jetson 侧 `scripts/` 代码，用于：

- 在线推理执行
- 真机数据采集
- 机器人观测构建
- AUBO 控制 helper 构建
- 多模态任务完成观测

默认部署形态如下：

```text
openpi/
├── repo/        # OpenPI 主仓库
├── scripts/     # 当前仓库
├── aubo_sdk/    # AUBO SDK
└── captures/    # 运行产物
```

## 仓库结构

```text
scripts/
├── config/
│   └── niic.env.example
├── tools/
│   ├── doctor.py
├── utils/
│   ├── env_utils.py
│   ├── path_utils.py
│   ├── pyorbbec_utils.py
│   └── runtime_config.py
├── support/
├── task/
├── data/
├── main.py
└── collect_data.py
```

`utils/` 放跨模块复用的环境、路径和通用工具。  
`support/` 放机器人控制、策略加载、观测构建、TUI 和 observer 等业务模块。

## 快速开始

1. 复制配置模板：

```bash
cp config/niic.env.example config/local.env
```

2. 修改 `config/local.env`

至少确认这些字段：

- `OPENPI_RUNTIME_PYTHON`
- `OPENPI_SDK_ROOT`
- `OPENPI_ROBOT_IP`
- `OPENPI_CHECKPOINT_DIR`
- `OPENPI_TASK_OBSERVER_PYTHON`
- `OPENPI_TASK_OBSERVER_MODEL`
- 串口相关配置

3. 运行环境检查：

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

4. 启动在线推理：

```bash
python3 main.py
```

5. 启动数据采集：

```bash
python3 collect_data.py
```

## 环境依赖

这个仓库不是完整的“单仓自足”环境，运行前还需要准备外部依赖。

### 目录依赖

默认要求以下目录与当前仓库并列存在：

- `../repo`
- `../aubo_sdk`

### Python 环境

当前推荐两套 Python：

- OpenPI 主环境  
  默认由 `OPENPI_RUNTIME_PYTHON` 指向，通常是 `../repo/.venv/bin/python`
- Observer / Gemma 环境  
  默认由 `OPENPI_TASK_OBSERVER_PYTHON` 指向

主流程默认复用 OpenPI 的 Python 环境。  
Gemma observer 单独使用一套环境，避免和 OpenPI / JAX 依赖互相污染。

### 系统和设备依赖

运行前通常还需要这些条件：

- Jetson + JetPack / CUDA
- AUBO SDK
- Orbbec Python SDK
- 串口设备
- Gemma 模型目录

`tools/doctor.py` 会尽量提前暴露这些缺项，但不会替代系统层安装。

## 主要入口

- `main.py`: 在线推理执行入口
- `collect_data.py`: 真机数据采集入口
- `support/load_policy.py`: 本地/远端策略统一加载
- `support/get_obs.py`: 双相机与机器人状态观测构建
- `support/task_observer.py`: 多模态任务完成判定

## PyTorch Local Backend

本地 PyTorch 推理现已支持通过独立 worker 进程运行：

- 主进程继续使用当前脚本环境执行相机、机器人与主控制逻辑
- PyTorch policy worker 单独运行在 `OPENPI_PYTORCH_RUNTIME_PYTHON` 指向的环境中
- worker 环境需要能加载 CUDA Torch、`transformers_replace` 和转换后的 `model.safetensors`

常用环境变量：

- `OPENPI_POLICY_BACKEND=pytorch`
- `OPENPI_PYTORCH_CHECKPOINT_DIR=.../model.safetensors 所在目录`
- `OPENPI_PYTORCH_RUNTIME_PYTHON=/home/niic/openpi/miniforge3/envs/openpi-py310-torch/bin/python`
- `OPENPI_PYTORCH_DEVICE=cuda`
- `OPENPI_SAMPLE_NUM_STEPS=5`

## 兼容说明

- `support/kubeconfig.yaml` 按敏感文件处理，默认不纳入 git
