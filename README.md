# OpenPI-AUBO i5 真机执行与数据采集系统
本仓库为 OpenPI 在 AUBO i5 机械臂上的 Jetson 侧完整系统，包含 **在线推理执行** 和 **训练数据采集** 两条链路。

## 快速开始
### 环境要求

- Jetson Orin + AUBO i5，机械臂控制柜网络可达
- 双路 Orbbec RGB 相机，主视角 + 腕部视角
- Lebai 夹爪，USB-RS485 通信，工具 IO 仅负责供电
- Python 3.10+，依赖包括 `numpy`、`opencv-python`、`pyorbbecsdk`、`pyserial`
- AUBO SDK 与本地 C++ helper 编译环境
- 可选末端六维力传感器，当前用于 `tcp_control` 中的向下运动力保护

### 运行推理

```bash
cd /home/orin/openpi/scripts

python main.py
```

启动后进入 TUI 菜单，可配置：
- **Frame**：`sim` / `real`
- **Policy**：`remote` / `local`
- **State Mode**：`j6` / `yaw`
- **Speed Mode**：`limited` / `native`
- **Exec Speed**：限速模式下的笛卡尔线速度，单位 m/s

### 运行数据采集

```bash
python collect_data.py
```

TUI 菜单可配置：
- **Task**：`pick_and_place` / `open_and_close`
- **Resume**：`continue` / `reset`
- **Save FPS**：`30` / `50`
- **State Mode**：`j6` / `yaw`
- **Mode**：`auto` / `manual`
- **Auto Episode**：`auto` 模式下的最大采集轮次

### 数据质检

```bash
python data/check_data.py data/pick_place
```

自动删除异常 episode 并重编号，输出 `dataset_health_report.json`。

---

## 如何训练出一个可用的权重

整体流程分为三步：**采集 -> 训练 -> 部署**。

### 第一步：采集训练数据

1. 在 Jetson 上运行 `collect_data.py`，选择目标任务
2. 系统自动完成：
   - **准备阶段**：恢复或生成当前工作区状态
   - **录制阶段**：执行任务的同时记录双路图像与机器人状态
   - **保存阶段**：以 30Hz 原始频率保存，或按配置重采样到 50Hz
   - **恢复阶段**：根据任务逻辑更新场景并进入下一轮
3. 采集结束后运行 `data/check_data.py` 清理异常数据
4. 每个任务建议采集 20 到 50 个 episode 起步

每个 episode 目录包含：`states.npy`、`actions.npy`、`timestamps.npy`、`env_steps.npy`、`images.npz`、`metadata.json`

### 第二步：在服务器端训练

将采集好的 `data/` 目录复制到训练机，使用 OpenPI 训练流程：

```bash
# 1. 转换数据到标准 LeRobot 格式
python convert_mujoco_to_lerobot.py

# 2. 在 openpi/repo 下确认训练配置
#    当前默认配置名：pi05_aubo_agv_lora

# 3. 计算归一化统计量
cd /path/to/openpi
source ./.venv/bin/activate
uv run scripts/compute_norm_stats.py --config-name pi05_aubo_agv_lora

# 4. 启动训练
uv run scripts/train.py --config pi05_aubo_agv_lora
```

### 第三步：部署推理

将训练好的 checkpoint 复制回 Jetson：

```bash
scp -r checkpoints/pi05_aubo_agv_lora/my_first_run/9999 \
    orin@172.18.10.44:/home/orin/openpi/repo/checkpoints/pi05_aubo_agv_lora/my_first_run/

python main.py
```

也可以部署为远端推理服务，由 Jetson 通过 WebSocket 调用。

---

## 文件结构与功能说明

```
scripts/
├── main.py                          # 在线推理主入口
├── collect_data.py                  # 真机数据采集主入口
├── support/
│   ├── get_obs.py                   # 观测构建，双相机 + 机器人状态
│   ├── gripper_control.py           # 夹爪串口控制
│   ├── joint_control.py             # 关节空间对齐与回原点
│   ├── load_policy.py               # 本地 / 远端策略统一加载
│   ├── pose_align.py                # sim / real 坐标系对齐
│   ├── tcp_control.py               # TCP servo / moveLine 控制与力保护
│   ├── force_sensor.py              # 六维力传感器串口驱动
│   └── tui_config.py                # main / collect_data 的 TUI 配置
├── task/
│   ├── pick_and_place.py            # pick / put / stack 任务规划
│   └── open_and_close.py            # open / close 任务规划与障碍物场景维护
├── data/
│   └── check_data.py                # 数据质检与坏样本清理
├── tcp_control_helper.cpp           # AUBO SDK TCP 常驻 helper
└── joint_control_helper.cpp         # AUBO SDK 关节控制 helper
```

### 各文件详细说明

#### `main.py` - 在线推理主入口

负责在线推理执行闭环：
- 加载策略
- 构建观测
- 推理动作
- 将动作积分为 TCP 轨迹
- 交由后台执行线程用 servo 模式执行

当前核心执行特点：
- 主线程负责观测、推理和动作分段
- 后台 `TrajectoryExecutor` 负责持续 servo 执行
- 支持 `yaw` / `j6` 两种观测模式
- 支持 `limited` / `native` 两种速度模式
- 夹爪状态变化会插入 TCP 前缀动作并等待夹爪到位
- 调用 `support/tcp_control.py` 时启用 TCP 最低高度保护与向下力保护

#### `collect_data.py` - 数据采集主入口

负责完整的真机采集流程组织：
- 启动 TUI 配置
- 读取 / 清理 `.collect_state.json`
- 初始化相机、夹爪和机器人
- 调用 task 规划模块生成任务步骤
- 执行动作并录制数据
- 保存 episode

当前实现特点：
- `pick_and_place` 维护 4 个物体的场景状态、堆叠关系和旋转信息
- `open_and_close` 维护 5 个障碍物场景、清障步骤和开关盒动作
- TCP 动作已统一改为 **servo streaming 执行**
- 录制时控制频率为 100Hz servo，图像和状态保存频率为 30Hz，保存前可选重采样到 50Hz
- 对所有向下 TCP 段启用 **实时 force guard**
- 向下动作在力传感器不可用或中途失效时会直接中止，优先保证安全
- 启动阶段的初始关节对齐仍使用 `moveJoint`

#### `support/get_obs.py` - 观测构建

并发获取双路 Orbbec 图像与机器人状态，统一裁剪 / 缩放到 224×224，并组装为 OpenPI 观测字典。  
被 `main.py` 和 `collect_data.py` 共同使用。

#### `support/gripper_control.py` - 夹爪控制

通过 USB-RS485 控制 Lebai 夹爪，提供：
- 开闭控制
- 状态读取
- 到位等待
- 稳态闭合判定

被 `main.py` 和 `collect_data.py` 调用。

#### `support/joint_control.py` - 关节控制

封装 `moveJoint`，主要用于：
- 启动后初始对齐
- 回原点
- 关节空间安全恢复

#### `support/load_policy.py` - 策略加载

提供统一的策略加载入口，屏蔽：
- 本地 checkpoint 推理
- 远端推理服务调用

只被 `main.py` 使用。

#### `support/pose_align.py` - 坐标系对齐

实现 sim 坐标系与真机坐标系之间的双向变换。  
被 `main.py`、`collect_data.py`、`support/tcp_control.py` 共同调用。

#### `support/tcp_control.py` - TCP 控制与力保护

当前是整个系统的核心运动控制模块，提供：
- `servo_pose` / `servo_pose_j6`
- `servo_chunk`
- 兼容保留的 `moveLine` 执行接口
- TCP z 最低高度裁剪
- 力传感器驱动接入
- 向下运动保护

当前 force guard 特点：
- 使用一阶低通滤波处理 `Fz`
- 采用 soft / hard 双阈值滞回
- `main.py` 的 servo 轨迹按 chunk 进入 live guard
- `collect_data.py` 可强制开启逐步 live force guard
- 向下力不足时逐步减小向下量，极端时禁止继续向下

#### `support/force_sensor.py` - 力传感器驱动

负责六维力传感器串口读数、帧解析、缓存与读取接口。  
当前被 `support/tcp_control.py` 用于实时安全约束。

#### `support/tui_config.py` - TUI 配置界面

提供终端交互式配置界面，被：
- `main.py`
- `collect_data.py`

共同使用。

#### `task/pick_and_place.py` - 抓放任务规划

负责：
- 维护 red / green / blue / apple 的场景状态
- 生成 `pick up ...`、`put ... on ...` 等任务 prompt
- 生成对应的 `TaskStep`
- 维护 `upper / lower / is_rotate / deg / standard_j6_rad`

只被 `collect_data.py` 调用。

#### `task/open_and_close.py` - 开关盒任务规划

负责：
- 维护 5 个障碍物的 `ObstacleScene`
- 随机生成清障带内 / 带外布局
- 生成清障步骤
- 生成 `open the storage box` / `close the storage box` 的动作序列

只被 `collect_data.py` 调用。

#### `data/check_data.py` - 数据质检

扫描 episode 目录，检查：
- 文件完整性
- `NaN / Inf`
- 四元数 / 轴角异常
- 时间轴与帧数一致性
- 零帧 / 重复帧 / 空图像

并支持自动删除坏样本与重编号。

#### `tcp_control_helper.cpp` / `joint_control_helper.cpp` - C++ helper

通过常驻进程方式封装 AUBO SDK，减少频繁建连开销，并为 Python 侧提供稳定的：
- snapshot
- servo
- moveJoint
- moveLine
- stop / wait

接口。

---

## 数据格式

每个 episode 目录包含：

| 文件 | 形状 | 说明 |
|------|------|------|
| `states.npy` | `(N, 7)` | `yaw` 模式：`[x, y, z, aa_x, aa_y, aa_z, gripper]` |
| `states.npy` | `(N, 8)` | `j6` 模式：`[x, y, z, aa_x, aa_y, aa_z, gripper, j6]` |
| `actions.npy` | `(N, 7)` | `yaw` 模式：`[dx, dy, dz, droll, dpitch, dyaw, gripper_next]` |
| `actions.npy` | `(N, 7)` | `j6` 模式：`[dx, dy, dz, droll, dpitch, dj6, gripper_next]` |
| `timestamps.npy` | `(N,)` | 目标保存频率下的时间戳 |
| `env_steps.npy` | `(N,)` | `0..N-1` |
| `images.npz` | - | 包含 `main_images` 与 `wrist_images` |
| `metadata.json` | - | 包含 `task/prompt`、`fps`、`state_mode`、`pose_frame` 等信息 |

---

## 调用关系

```
main.py
├── support/tui_config.py
├── support/load_policy.py
├── support/get_obs.py
├── support/pose_align.py
├── support/tcp_control.py
│   ├── support/force_sensor.py
│   └── tcp_control_helper.cpp
├── support/gripper_control.py
└── support/joint_control.py
    └── joint_control_helper.cpp

collect_data.py
├── task/pick_and_place.py
├── task/open_and_close.py
├── support/get_obs.py
├── support/pose_align.py
├── support/tcp_control.py
│   ├── support/force_sensor.py
│   └── tcp_control_helper.cpp
├── support/gripper_control.py
└── support/joint_control.py

data/check_data.py
└── 独立运行，对 episode 目录做质检和清理
```
