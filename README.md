# OpenPI Jetson 侧机器人执行与数据采集系统说明

## 1. 文档目的

本文档用于说明 `/home/orin/openpi/scripts` 当前的系统能力、运行方式、数据格式与核心实现，作为 Jetson 侧真实机器人执行与采集系统的对外汇报材料。

文档面向项目负责人、管理者和后续维护人员，重点回答以下问题：

- 当前系统能完成什么
- 系统如何运行
- 采集数据以什么格式保存
- 关键模块分别承担什么职责
- 当前仍需重点关注哪些工程配置项


## 2. 系统概述

`scripts` 目录承担 OpenPI 在 Jetson Orin 端的机器人执行层与数据采集层，围绕 AUBO i5 机械臂、双路 Orbbec 相机与夹爪构成完整闭环。

当前系统覆盖的主链路包括：

- 双相机观测采集与预处理
- 机器人状态读取与观测组织
- OpenPI 策略加载与推理
- TCP 笛卡尔轨迹规划与执行
- 夹爪开合控制与状态管理
- 真机 pick / put / stack 数据采集
- episode 落盘、断点续采与数据质检

系统当前已经形成统一的运行接口和统一的数据产出格式，可同时支撑在线执行与离线训练数据生产。


## 3. 当前功能能力

### 3.1 实时推理执行

系统支持在 Jetson 上加载 OpenPI 策略并驱动真实机械臂执行。

当前策略接入方式包括：

- 本地 checkpoint 直接推理
- 远端推理服务经 websocket 调用

实时执行链路如下：

1. 采集双路相机图像与机器人状态
2. 组织成 OpenPI 标准 observation
3. 策略输出 TCP delta 与 gripper action
4. 在仿真坐标系中积分目标位姿
5. 映射到真实机器人坐标系
6. 进行边界约束、逆解与轨迹合法性检查
7. 通过 `moveLine` 轨迹块连续执行
8. 根据动作结果更新夹爪状态与执行状态

该链路的主入口为：

- [main.py](/home/orin/openpi/scripts/main.py)


### 3.2 真机数据采集

系统支持在真实环境下自动采集 OpenPI 训练数据，当前支持 `pick`、`put` 与 `stack` 三类任务。

任务生成方式如下：

- `pick`: 自动生成 `pick up the <color> cube`
- `put`: 自动生成 `put the <color1> cube on the <color2> cube`
- `stack`: 自动生成 `stack <bottom> <middle> <top> cubes from bottom to top`

当前采集能力包括：

- 三色方块位置维护
- 准备阶段自动整理工作区
- 按颜色、颜色对或三元组顺序持续采集
- 手动启动或自动连续执行
- 中断状态持久化与恢复
- 原始 `30Hz` 观测记录
- 保存前重采样到 `50Hz`
- 保存格式与 WSL / MuJoCo 侧脚本保持一致
- 任务结束后自动执行整理与恢复工作区

当前采集主入口为：

- [collect_data.py](/home/orin/openpi/scripts/collect_data.py)


### 3.3 数据质量检查

系统提供数据集健康检查脚本，对 `data/episode_*` 自动执行结构检查、数值检查与图像健康检查。

当前检测项包括：

- 必需文件完整性
- `states / actions / timestamps / env_steps / images / metadata` 的 shape 与 dtype
- `NaN / Inf`
- 四元数范数异常
- `actions` 与相邻 `states` 差分一致性
- 时间轴与 `50Hz` 网格一致性
- 黑帧
- 唯一帧数过低
- 长时间连续重复帧

当前脚本默认行为包括：

- 自动识别异常 episode
- 自动删除异常 episode
- 自动对剩余 episode 重编号
- 生成 JSON 质检报告

脚本路径：

- [check_dataset_health.py](/home/orin/openpi/scripts/data/check_dataset_health.py)


### 3.4 坐标系对齐

系统实现了从仿真坐标系到真实机器人坐标系的位姿对齐模块，用于连接 MuJoCo 侧策略输出和真实机器人执行层。

当前策略为：

- 位置映射采用 yaw 旋转加平移
- 姿态映射采用旋转矩阵组合

该模块负责保证 sim 侧 delta 能以符合真实机器人语义的方式落到真实 TCP 控制链路中。

实现文件：

- [pose_align.py](/home/orin/openpi/scripts/pose_align.py)


### 3.5 相机观测构建

系统当前支持双路 Orbbec RGB 观测：

- 主视角相机
- 腕部相机

观测构建内容包括：

- 双路图像抓取
- 图像 resize / pad 到 OpenPI 输入规格
- TCP pose 读取
- 夹爪状态拼接
- observation dict 组装

实时推理使用：

- [get_obs.py](/home/orin/openpi/scripts/get_obs.py)

数据采集使用：

- [collect_data.py](/home/orin/openpi/scripts/collect_data.py) 中的 `CameraPair`


### 3.6 夹爪控制

夹爪通过 USB-RS485 控制，当前支持：

- 打开 / 闭合
- 状态读取
- 到位等待
- 稳态闭合判定
- OpenPI `0 / 1` 语义对齐

当前默认参数：

- 默认速度：`20`
- 默认力度：`10`

实现文件：

- [gripper_control.py](/home/orin/openpi/scripts/gripper_control.py)


## 4. 运行方式与业务流程

### 4.1 在线推理流程

在线推理由 [main.py](/home/orin/openpi/scripts/main.py) 负责调度，当前运行方式如下：

1. 编译并加载 C++ helper
2. 加载策略
3. 初始化相机与控制对象
4. 循环构造 observation
5. 执行策略推理
6. 将动作提交到 TCP 执行线程
7. 在循环中处理夹爪动作与执行状态
8. 输出每轮耗时、跟踪结果与采样信息

当前主循环与运动执行线程解耦：

- 主线程负责观测与推理
- 后台执行器负责连续提交与等待运动轨迹

该结构用于保证策略推理和真实运动可以并行推进。


### 4.2 数据采集流程

数据采集由 [collect_data.py](/home/orin/openpi/scripts/collect_data.py) 负责，当前业务流程如下：

1. 读取断点状态文件 `.collect_state.json`
2. 若存在历史状态，则选择继续或清空后重置
3. 若为新采集，则进入准备阶段整理三色方块
4. 维护三色方块当前 `xy` 坐标
5. 根据当前任务选择颜色、颜色对或三色顺序并自动生成 prompt
6. 返回原点后开始录制 episode
7. 在执行段内同步记录机器人状态、夹爪状态与双路图像
8. 任务结束后执行对应的整理动作并恢复工作区状态
9. 将原始 `30Hz` 序列重采样到 `50Hz`
10. 以统一格式落盘
11. 更新状态文件并进入下一轮

当前放置逻辑满足以下要求：

- 三个方块随机落点间隔不小于约 `0.08m`
- 每轮任务结束后先完成放置，再回到原点
- 录制只覆盖任务执行段，不包含准备段和收尾段
- `stack` 任务在录制后会执行拆堆与随机回放，保证下一轮仍从分散状态开始


### 4.3 数据质检流程

数据质检由 [check_dataset_health.py](/home/orin/openpi/scripts/data/check_dataset_health.py) 负责，当前流程如下：

1. 扫描 `data/episode_*`
2. 逐个读取数组与 metadata
3. 执行结构、数值与图像检查
4. 对异常样本标记结果
5. 默认删除异常样本
6. 重排剩余 episode 编号
7. 输出 `dataset_health_report.json`


## 5. 数据格式说明

当前采集结果与 WSL 侧训练脚本使用统一格式，单个 episode 目录包含以下文件：

- `states.npy`
- `actions.npy`
- `timestamps.npy`
- `env_steps.npy`
- `images.npz`
- `metadata.json`


### 5.1 `states.npy`

形状：

- `(N, 8)`

定义：

- `[x, y, z, qw, qx, qy, qz, gripper]`

说明：

- 位置为仿真坐标系下的位置状态
- 姿态采用 `wxyz` 四元数
- `gripper` 采用 `0 / 1` 标量


### 5.2 `actions.npy`

形状：

- `(N, 7)`

定义：

- `[dx, dy, dz, droll, dpitch, dyaw, gripper_next]`

说明：

- 由相邻 state 自动差分得到
- 最后一帧位移与角位移补零
- 最后一帧夹爪值继承末状态


### 5.3 `timestamps.npy` 与 `env_steps.npy`

当前保存规则如下：

- `env_steps = 0..N-1`
- `timestamps = env_steps / 50`

因此落盘后的时间轴严格对应 `50Hz` 采样网格。


### 5.4 `images.npz`

包含：

- `main_images`
- `wrist_images`

图像规格：

- `(N, 224, 224, 3)`
- `uint8`


## 6. 核心模块说明

### 6.1 策略加载层

文件：

- [load_policy.py](/home/orin/openpi/scripts/load_policy.py)

职责：

- 提供统一的策略加载入口
- 屏蔽本地推理与远端推理的接入差异
- 对上层暴露统一 `infer / reset / metadata` 接口

核心对象：

- `PolicyLoadSpec`
- `LocalPolicyRunner`
- `RemotePolicyRunner`
- `load_policy()`


### 6.2 观测构建层

文件：

- [get_obs.py](/home/orin/openpi/scripts/get_obs.py)

职责：

- 并发抓取图像与机器人状态
- 组织 OpenPI observation
- 对图像进行对齐、裁剪与格式整理

核心对象与函数：

- `RealRobotOpenPIObservationBuilder`
- `_capture_best_pair()`
- `pose6_to_openpi_state()`

设计特性：

- 使用线程池并发读取相机与 robot snapshot
- 夹爪状态以本地缓存为主，并在控制更新失败时读回校正


### 6.3 TCP 控制层

文件：

- [tcp_control.py](/home/orin/openpi/scripts/tcp_control.py)
- [tcp_control_helper.cpp](/home/orin/openpi/scripts/tcp_control_helper.cpp)

职责：

- 将策略输出的 TCP delta 转换为真实机器人轨迹
- 完成边界约束、逆解校验与轨迹合法性检查
- 调用 SDK 的 `moveLine` 接口执行真实笛卡尔运动

核心对象与函数：

- `RobotSnapshot`
- `_DaemonHelper`
- `plan_tcp_action_chunk_movel()`
- `execute_tcp_action_chunk()`

当前控制特性：

- 采用 `moveLine` 分块执行
- 支持非阻塞轨迹提交
- 支持 chunk 内多 waypoint 与 blend 过渡
- 执行完成后以真实停点回填结果
- 对 waypoint 间扫掠路径执行采样检查
- 真实笛卡尔速度可通过 `speed_mps` 显式设置


### 6.4 C++ Helper 层

文件：

- [tcp_control_helper.cpp](/home/orin/openpi/scripts/tcp_control_helper.cpp)
- [joint_control_helper.cpp](/home/orin/openpi/scripts/joint_control_helper.cpp)

职责：

- 与 AUBO SDK 保持常驻连接
- 下沉实时控制与状态读取逻辑
- 对 Python 暴露稳定的命令式接口

当前对外命令包括：

- `snapshot`
- `movel`
- `movel_chunk`
- `wait_motion`
- `stop_motion`
- `move_joint`

设计目标：

- 降低重复连接开销
- 降低 Python 高频实时控制的不确定性
- 将关键控制时序留在 C++ 侧完成


### 6.5 关节控制层

文件：

- [joint_control.py](/home/orin/openpi/scripts/joint_control.py)
- [joint_control_helper.cpp](/home/orin/openpi/scripts/joint_control_helper.cpp)

职责：

- 提供 `moveJoint` 控制能力
- 用于初始位姿对齐、回原点与准备阶段姿态恢复

返回结果中包含：

- `current_q`
- `target_q`
- `final_q`
- `err`
- `collision`


### 6.6 夹爪控制层

文件：

- [gripper_control.py](/home/orin/openpi/scripts/gripper_control.py)

职责：

- 串口通信
- 开合控制
- 读回状态
- 到位等待
- 稳态闭合判定

核心对象与函数：

- `GripperController`
- `command_gripper_state()`
- `is_gripper_stably_closed()`


### 6.7 数据采集层

文件：

- [collect_data.py](/home/orin/openpi/scripts/collect_data.py)

职责：

- 组织真机 pick / put / stack 采集任务
- 维护三色方块状态
- 执行采样、重采样与落盘
- 处理状态续采与相机重建

核心对象与函数：

- `CameraPair`
- `execute_and_record()`
- `resample_episode_to_50hz()`
- `save_episode()`
- `save_collect_state()`
- `load_collect_state()`

当前状态文件：

- `.collect_state.json`

保存内容包括：

- 三色方块当前 `xy`
- 当前任务索引
- 当前颜色索引
- 已采 episode 数


### 6.8 数据质检层

文件：

- [check_dataset_health.py](/home/orin/openpi/scripts/data/check_dataset_health.py)

职责：

- 自动筛查异常 episode
- 自动删除异常目录
- 自动整理编号
- 输出质检结果报告


## 7. 当前配置与运行特性

当前系统的重要运行参数如下：

- 采集原始帧率：`30Hz`
- 保存帧率：`50Hz`
- 默认 `moveLine` 笛卡尔速度：`0.10 m/s`
- 方块最小安全间距：`0.08m`
- 默认夹爪力度：`10`
- 默认夹爪速度：`20`

当前运行方式具备以下工程特征：

- 策略加载入口统一
- 在线推理与运动执行解耦
- 采集格式与训练侧统一
- 状态文件支持断点续采
- 质检脚本支持自动清理异常样本


## 8. 当前关注项与建议

当前仍建议持续关注以下工程项：

- 碰撞检测阈值与停机策略仍建议按任务场景分别标定并形成固定配置
- 采集流程、质检流程与数据存储策略可以进一步做成一键化作业链路
- 运行参数建议继续沉淀为集中配置，便于不同场景下快速切换


## 9. 结论

当前 `/home/orin/openpi/scripts` 已形成完整的 Jetson 侧真实机器人执行与数据采集系统，能够支撑真实机器人在线推理执行、三色方块任务数据采集、断点续采、统一格式落盘以及自动数据质检。

从对外汇报角度看，当前系统已经具备以下交付特征：

- 真实机器人执行链路完整
- 训练数据生产链路完整
- 数据格式与训练侧对齐
- 运行、采集、质检职责边界清晰
- 核心模块结构稳定，便于继续扩展任务类型和工程配置
