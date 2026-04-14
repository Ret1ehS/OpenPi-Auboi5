# OpenPI-AUBO i5 Scripts

这个仓库只跟踪 `scripts/` 侧代码，默认部署形态如下：

```text
openpi/
├── repo/        # OpenPI 主仓库与训练/推理依赖
├── scripts/     # 当前仓库
├── aubo_sdk/    # AUBO SDK
└── captures/    # 运行产物
```

## 当前结构

```text
scripts/
├── config/
│   └── niic.env.example
├── tools/
│   ├── doctor.py
│   ├── build_helpers.sh
│   ├── setup_runtime.sh
│   ├── setup_observer_env.sh
│   ├── run_main.sh
│   └── run_collect.sh
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

`utils/` 放跨模块复用的路径、环境和相机格式转换逻辑。  
`support/` 保留机器人控制、观测构建、策略加载、TUI 和 task observer 这类业务模块。

## 配置方式

1. 复制示例配置：

```bash
cp config/niic.env.example config/local.env
```

2. 按机器实际情况修改 `config/local.env`，至少确认这些字段：

- `OPENPI_SDK_ROOT`
- `OPENPI_ROBOT_IP`
- `OPENPI_CHECKPOINT_DIR`
- `OPENPI_TASK_OBSERVER_PYTHON`
- `OPENPI_TASK_OBSERVER_MODEL`
- 串口相关配置

说明：

- Python 代码会自动尝试加载 `config/local.env` 或 `config/niic.env`
- shell 脚本也会优先 source `config/local.env`
- 也可以显式指定 `OPENPI_ENV_FILE=/path/to/xxx.env`

## 推荐使用流程

先做环境体检：

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

编译 helper：

```bash
bash tools/build_helpers.sh
```

主运行环境准备：

```bash
bash tools/setup_runtime.sh
```

observer 环境检查：

```bash
bash tools/setup_observer_env.sh
```

启动在线推理：

```bash
bash tools/run_main.sh
```

启动数据采集：

```bash
bash tools/run_collect.sh
```

## 说明

这次整理的目标是把“机器相关配置”和“可复用工具”从业务逻辑里抽出来，不是把 Jetson、CUDA、AUBO SDK、相机 SDK 做成全自动裸机安装器。

原因很简单：这套环境依赖硬件、驱动、JetPack、厂商 SDK 和本地目录结构，写成一个全自动脚本会非常脆。当前仓库更适合：

- 用 `config/*.env` 收口机器配置
- 用 `tools/doctor.py` 提前暴露缺项
- 用 `tools/*.sh` 固化常用启动与构建入口

## 主要模块

- `main.py`: 在线推理执行入口，已接入 task observer 软停止逻辑
- `collect_data.py`: 真机数据采集入口
- `support/load_policy.py`: 本地/远端策略统一加载
- `support/get_obs.py`: 双相机与机器人状态观测构建
- `support/task_observer.py`: 多模态任务完成判定
- `utils/runtime_config.py`: 默认配置和环境变量收口
- `utils/path_utils.py`: `repo/.build/captures/log/aubo_sdk` 路径推导

## 注意

- `support/path_utils.py` 和 `support/pyorbbec_utils.py` 现在只是兼容层，新的代码应直接从 `utils/` 导入
- `support/kubeconfig.yaml` 仍按敏感文件处理，默认不纳入 git
