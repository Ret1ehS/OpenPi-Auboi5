# OpenPI-AUBO i5 真机执行与数据采集系统

本仓库为 OpenPI 在 AUBO i5 机械臂上的 Jetson 侧完整系统，包含 **在线推理执行** 和 **训练数据采集** 两条链路。

## 快速开始

### 环境要求

- Jetson Orin + AUBO i5（IP: 192.168.1.100）
- 双路 Orbbec RGB 相机（主视角335L + 腕部305）
- Lebai 夹爪（USB-RS485通信 机械臂末端工具IO供电）
- Python 3.10+，依赖：`numpy`, `cv2`, `pyorbbecsdk`, `pyserial` 等
- AUBO SDK (`libaubo_sdk.so`)，C++ helper 会在首次运行时自动编译

### 运行推理

```bash
cd /home/orin/openpi/scripts

# checkpoint 推理
python main.py

```

启动后进入 TUI 菜单，可配置:
- **Frame**：`sim` / `real`
- **Policy**：`remote` / `local`
- **State Mode**：`j6`（不锁 yaw）/ `yaw`（锁 yaw）
- **Speed Mode**：`limited`（限速）/ `native`（原生速度，自动插值）
- **Exec Speed**：限速模式下的笛卡尔速度（m/s）

### 运行数据采集

```bash
# 采集任务数据
python collect_data.py
```

- **Task**：`pick_and_place` / `open_and_close` / `后续任务`
- **Resume**：`conntinue` / `reset`
- **Save FPS**：`30` / `50`
- **State Mode**：`j6` / `yaw`
- **Mode**：`auto` / `manual`
- **Auto Episode**：Mode-auto时可选最大采集轮次


### 数据质检

```bash
python data/check_data.py data/pick_place
```

自动删除异常 episode 并重编号，输出 `dataset_health_report.json`。

---

## 如何训练出一个可用的权重

整体流程分为三步：**采集 → 训练 → 部署**。

### 第一步：采集训练数据(硬编码位姿)

1. 在 Jetson 上运行 `collect_data.py`，选择目标任务（`pick_and_place` 或 `open_and_close`）
2. 系统自动完成：
   - **准备阶段**：将物体从原点摆放到随机位置
   - **录制阶段**：执行任务的同时以 30Hz 记录双路图像 + 机器人状态
   - **保存阶段(可选)**：重采样到 50Hz，以标准格式落盘
   - **恢复阶段**：整理工作区，进入下一轮
3. 采集结束后运行 `data/check_data.py` 清理异常数据
4. 每个任务建议采集 **20-50 个 episode**

采集产出的每个 episode 包含：`states.npy`, `actions.npy`, `timestamps.npy`, `env_steps.npy`, `images.npz`, `metadata.json`

### 第二步：在服务器端训练

将采集好的 `data/` 目录拷贝到训练机器，使用 OpenPI 官方训练流程：

```bash
# 1.转换数据到标准LeRobot格式(见mujoco-env仓库)
python convert_mujoco_to_lerobot.py

# 在 openpi/repo 目录下
# 2. 确认训练配置（config 中指定数据路径和 LoRA 参数）
#    默认配置名: pi05_aubo_agv_lora

# 3.数据归一化
cd /path/to/openpi
source ./.venv/bin/activate
uv run scripts/compute_norm_stats.py --config-name pi05_aubo_agv_lora

# 4. 启动训练
uv run scripts/train.py --config pi05_aubo_agv_lora

# 5. 训练完成后在 checkpoints/ 下生成权重
#    例如: checkpoints/pi05_aubo_agv_lora/my_first_run/9999
```

### 第三步：部署推理

将训练好的 checkpoint 拷贝到 Jetson：

```bash
# 本地推理：将 checkpoint 放到默认路径
scp -r checkpoints/pi05_aubo_agv_lora/my_first_run/9999 \
    orin@172.18.10.44:/home/orin/openpi/repo/checkpoints/pi05_aubo_agv_lora/my_first_run/

# 运行
python main.py
```

或使用远端推理服务（K8s pod 部署 checkpoint，Jetson 基于 Kubeconfig.yaml 通过 WebSocket 调用:

---

## 文件结构与功能说明

```
scripts/
├── main.py                          # 在线推理主入口
├── collect_data.py                  # 数据采集主入口
├── support/
│   ├── get_obs.py                   # 观测构建（相机 + 机器人状态）
│   ├── gripper_control.py           # 夹爪串口控制
│   ├── joint_control.py             # 关节空间控制
│   ├── load_policy.py               # 策略加载（本地/远端）
│   ├── pose_align.py                # 仿真-真机坐标系对齐
│   ├── tcp_control.py               # TCP 笛卡尔轨迹规划与执行
│   └── tui_config.py                # TUI 交互式配置菜单
├── task/
│   ├── pick_and_place.py            # pick/put 任务规划
│   └── open_and_close.py            # 开关盖任务规划 + 障碍物管理
├── data/
│   └── check_data.py                # 数据集质检与清理
├── tcp_control_helper.cpp           # AUBO SDK C++ 控制守护进程
└── joint_control_helper.cpp         # AUBO SDK C++ 关节控制守护进程
```

### 各文件详细说明

#### `main.py` — 在线推理主入口

负责整个推理执行循环：加载策略 → 构建观测 → 推理输出动作 → 积分轨迹 → 执行运动。

核心逻辑：
- 主线程串行执行：观测采集 → 策略推理 → 动作积分，每次取 10 个 interval
- 后台 `TrajectoryExecutor` 线程异步执行 servo 轨迹
- 支持 limited（限速重定时）和 native（原生速度，10→50 线性插值）两种速度模式
- 夹爪状态变化时执行 TCP 前缀段，等待夹爪到位后重新推理
- z_clip 安全机制：TCP 最低点不低于 180mm

**调用的模块**：
- `support/load_policy.py` → 加载策略
- `support/get_obs.py` → 构建观测
- `support/tcp_control.py` → 轨迹规划 + 执行
- `support/pose_align.py` → 坐标系转换
- `support/gripper_control.py` → 夹爪控制
- `support/joint_control.py` → 回原点
- `support/tui_config.py` → 启动前配置

#### `collect_data.py` — 数据采集主入口

负责组织完整的采集流程：准备工作区 → 规划任务 → 录制执行 → 保存数据 → 恢复工作区。

核心逻辑：
- **pick_and_place 模式**：维护 4 个物体（red/green/blue/apple）的场景状态，自动生成 pick/put/stack 任务，录制执行段，任务后执行 post_steps 恢复场景
- **open_and_close 模式**：维护 5 个障碍物的场景状态，每个周期随机布局 → 物理摆放 → 清障（录制，属于 open episode）→ 开盖（录制）→ 关盖（录制），每周期产出 2 个 episode
- 30Hz 原始录制，保存前重采样到 50Hz(可选)
- 支持断点续采（`.collect_state.json`）
- 支持连续采集和 `--dry-run` 空跑

**调用的模块**：
- `task/pick_and_place.py` → 任务规划（pick/put/stack 步骤生成）
- `task/open_and_close.py` → 任务规划（开关盖步骤 + 障碍物布局 + 清障步骤）
- `support/tcp_control.py` → 轨迹执行
- `support/pose_align.py` → 坐标系转换
- `support/gripper_control.py` → 夹爪控制
- `support/joint_control.py` → 回原点
- `support/get_obs.py` → 录制时的相机采集（使用 `CameraPair`）

#### `support/get_obs.py` — 观测构建

并发采集双路 Orbbec 相机图像 + 机器人 TCP 状态，组装成 OpenPI 标准 observation dict。图像 resize/pad 到 224×224。

被 `main.py`（推理观测）和 `collect_data.py`（录制帧）调用。

#### `support/gripper_control.py` — 夹爪控制

通过 USB-RS485 串口控制 Lebai 夹爪，支持开/闭、状态读取、到位等待、稳态闭合判定。OpenPI 0/1 语义对齐。

被 `main.py` 和 `collect_data.py` 调用。

#### `support/joint_control.py` — 关节空间控制

封装 `moveJoint` 命令，用于回原点、初始位姿对齐。通过 C++ helper 进程与 AUBO SDK 通信。

被 `main.py`（回原点）和 `collect_data.py`（准备阶段、每轮回原点）调用。

#### `support/load_policy.py` — 策略加载

统一的策略加载入口，屏蔽本地推理（直接加载 checkpoint）和远端推理（WebSocket + kubectl port-forward）的差异。

仅被 `main.py` 调用。

#### `support/pose_align.py` — 坐标系对齐

实现仿真坐标系（MuJoCo/OpenPI）与真实机器人坐标系之间的双向转换。位置映射采用 yaw 旋转 + 平移，姿态映射采用旋转矩阵组合。

被 `main.py`、`collect_data.py`、`tcp_control.py` 调用，贯穿整个系统。

#### `support/tcp_control.py` — TCP 轨迹规划与执行

核心控制模块。将策略输出的 TCP delta 在仿真坐标系中积分，转换到真实坐标系，经边界约束、逆解校验、碰撞检查后，通过 `moveLine` 分块执行。支持 z_clip 安全限制（180mm 最低点）。

被 `main.py`（推理轨迹执行）和 `collect_data.py`（采集任务执行）调用。

#### `support/tui_config.py` — TUI 配置菜单

终端交互式配置界面，支持方向键导航、选项切换。配置项包括任务类型、状态模式、速度模式、执行速度等。

被 `main.py` 在启动时调用。

#### `task/pick_and_place.py` — 抓放任务规划

为 pick/put/stack 任务生成执行步骤序列。维护 4 个物体（3 色方块 + apple）的场景状态（位置、旋转、堆叠关系），自动生成任务 prompt 和 pick/place 步骤。

仅被 `collect_data.py` 调用。

#### `task/open_and_close.py` — 开关盖任务规划

为 open/close 任务生成执行步骤。包含：
- 5 个障碍物的场景管理（`ObstacleScene`），支持堆叠
- 随机布局生成（2-4 个落在目标 y 带内，70% 概率堆叠 2 个）
- 清障步骤生成（将带内障碍物移到带外，堆叠的先拆上层）
- 开盖/关盖动作步骤生成

仅被 `collect_data.py` 调用。

#### `data/check_data.py` — 数据质检

扫描 episode 目录，检查文件完整性、数值异常（NaN/Inf）、四元数范数、时间轴对齐、黑帧、重复帧等。自动删除异常 episode 并重编号。

独立运行，不被其他文件调用。

#### `tcp_control_helper.cpp` / `joint_control_helper.cpp` — C++ 守护进程

与 AUBO SDK 保持常驻连接，接收 Python 侧的控制命令（snapshot、movel、move_joint 等），降低重复连接开销和 Python 高频控制的不确定性。

由 `tcp_control.py` 和 `joint_control.py` 自动编译并启动。

---

## 数据格式

每个 episode 目录（`episode_XXXXXX/`）包含：

| 文件 | 形状 | 说明 |
|------|------|------|
| `states.npy` | `(N, 7)` | `[x, y, z, aa_x, aa_y, aa_z, gripper]`，仿真坐标系, yaw格式 |
| `states.npy` | `(N, 8)` | `[x, y, z, aa_x, aa_y, aa_z, gripper, j6]`，仿真坐标系, j6格式 |
| `actions.npy` | `(N, 7)` | `[dx, dy, dz, droll, dpitch, dyaw, gripper]`，相邻 state 差分, yaw格式 |
| `actions.npy` | `(N, 7)` | `[dx, dy, dz, droll, dpitch, dj6, gripper]`，相邻 state 差分， j6格式 |
| `timestamps.npy` | `(N,)` | `env_steps / 50/30`，50Hz/30Hz 网格 |
| `env_steps.npy` | `(N,)` | `0..N-1` |
| `images.npz` | — | 含 `main_images (N,224,224,3)` 和 `wrist_images (N,224,224,3)` |
| `metadata.json` | — | 含 `prompt`, `save_fps`, `state_mode` 等 |

---

## 调用关系

```
main.py (在线推理)
├── support/tui_config.py        ← 启动配置
├── support/load_policy.py       ← 加载策略
├── support/get_obs.py           ← 构建观测
├── support/pose_align.py        ← 坐标系转换
├── support/tcp_control.py       ← 轨迹规划执行
│   └── tcp_control_helper.cpp   ← C++ SDK 通信
├── support/gripper_control.py   ← 夹爪控制
└── support/joint_control.py     ← 回原点
    └── joint_control_helper.cpp ← C++ SDK 通信

collect_data.py (数据采集)
├── task/pick_and_place.py       ← pick/put/stack 任务规划
├── task/open_and_close.py       ← 开关盖任务规划 + 障碍物管理
├── support/get_obs.py           ← 录制时相机采集
├── support/pose_align.py        ← 坐标系转换
├── support/tcp_control.py       ← 轨迹执行
├── support/gripper_control.py   ← 夹爪控制
└── support/joint_control.py     ← 回原点

data/check_data.py (独立运行)
└── 扫描 episode 目录，质检 + 清理
```
