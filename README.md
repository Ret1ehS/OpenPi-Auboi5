# OpenPI-AUBO i5

这是 AUBO i5 机器人侧的 OpenPI 运行脚本仓库。

本仓库只放机器人运行、观测、策略加载和数据采集相关代码。OpenPI 训练仓库、AUBO SDK、checkpoint、模型和设备驱动都作为外部依赖，通过 `config.yaml` 配置。

## 目录

- `main.py`: 在线推理入口。
- `data/collect_data.py`: 真机数据采集。
- `data/check_data.py`: episode 数据检查。
- `data/convert_data.py`: 数据转换工具。
- `support/get_obs.py`: 相机采集和 OpenPI observation 构建。
- `support/load_policy.py`: 本地或远程策略加载。
- `support/gripper_control.py`: 夹爪串口控制。
- `support/task_observer.py`: 可选的任务完成判断。
- `task/`: 任务工具和 observer 判断规则。
- `tools/doctor.py`: 环境和配置检查。

## 配置

默认读取根目录 `config.yaml`。

如果要使用另一份本机私有配置：

```bash
export OPENPI_CONFIG_FILE=/path/to/config.yaml
```

通常需要确认这些字段：

- `OPENPI_RUNTIME_PYTHON`
- `OPENPI_SDK_ROOT`
- `OPENPI_ROBOT_IP`
- `OPENPI_CHECKPOINT_DIR`
- 夹爪和力传感器串口
- 策略后端和 checkpoint 路径

旧的 `OPENPI_ENV_FILE` 仍然兼容，但新配置建议使用 `config.yaml`。

## 快速开始

检查环境：

```bash
python3 tools/doctor.py
```

运行在线推理：

```bash
python3 main.py
```

采集真机数据：

```bash
python3 data/collect_data.py
```

检查采集数据：

```bash
python3 data/check_data.py /path/to/dataset
```

## Task Observer

Task observer 是可选功能，不是运行必须项。即使关闭 observer，机器人仍然可以正常推理和采集数据。

只有在需要系统根据相机图像和机器人状态周期性判断任务是否完成时，才启用它：

```yaml
env:
  OPENPI_TASK_OBSERVER_ENABLE: true
  OPENPI_TASK_OBSERVER_PYTHON: /path/to/observer/python
  OPENPI_TASK_OBSERVER_MODEL: /path/to/gemma/model
  OPENPI_TASK_OBSERVER_SPEC_FILE: task/prompt.txt
```

observer 的任务完成规则写在 `task/prompt.txt`。

如果 observer 环境或模型还没配好，保持关闭即可，先跑通主流程。

## 策略模式

远程策略：

```yaml
env:
  OPENPI_POLICY_BACKEND: auto
  OPENPI_K8S_NAMESPACE: ...
  OPENPI_POLICY_LOCAL_PORT: 8000
  OPENPI_POLICY_REMOTE_PORT: 8000
```

本地 PyTorch 策略：

```yaml
env:
  OPENPI_POLICY_BACKEND: pytorch
  OPENPI_PYTORCH_RUNTIME_PYTHON: /path/to/torch/python
  OPENPI_PYTORCH_CHECKPOINT_DIR: /path/to/pytorch/checkpoint
  OPENPI_PYTORCH_DEVICE: cuda
```

## 备注

- `support/kubeconfig.yaml` 属于本机敏感配置，不应提交。
- `docs/` 和 `tests/` 默认只作为本地目录使用。
- 只检查主运行环境：`python3 tools/doctor.py --section runtime`。
- 只检查 observer 环境：`python3 tools/doctor.py --section observer`。
