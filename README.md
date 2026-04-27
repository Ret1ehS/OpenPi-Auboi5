# OpenPI-AUBO i5

Robot-side scripts for running OpenPI on an AUBO i5 robot.

This repository contains only the robot/runtime glue code. The OpenPI training
repository, AUBO SDK, checkpoints, models, and device drivers are external
dependencies configured through `config.yaml`.

## What Is Here

- `main.py`: online inference entrypoint.
- `data/collect_data.py`: real-robot data collection.
- `data/check_data.py`: collected episode inspection.
- `data/convert_data.py`: dataset conversion helpers.
- `support/get_obs.py`: camera capture and OpenPI observation building.
- `support/load_policy.py`: local or remote policy loading.
- `support/gripper_control.py`: gripper serial control.
- `support/task_observer.py`: optional task-completion observer.
- `task/`: task utilities and observer completion rules.
- `tools/doctor.py`: environment/config sanity checks.

## Configuration

Runtime config is loaded from root `config.yaml` by default.

Use a private config file when running on another machine:

```bash
export OPENPI_CONFIG_FILE=/path/to/config.yaml
```

Common fields to check:

- `OPENPI_RUNTIME_PYTHON`
- `OPENPI_SDK_ROOT`
- `OPENPI_ROBOT_IP`
- `OPENPI_CHECKPOINT_DIR`
- serial ports for the gripper and force sensor
- policy backend/checkpoint settings

`OPENPI_ENV_FILE` is still accepted for old env-style files, but new setups
should use `config.yaml`.

## Quick Start

Check the environment:

```bash
python3 tools/doctor.py
```

Run online inference:

```bash
python3 main.py
```

Collect real-robot data:

```bash
python3 data/collect_data.py
```

Check collected data:

```bash
python3 data/check_data.py /path/to/dataset
```

## Task Observer

The task observer is optional. The robot can run inference and collect data
without it.

Enable it only when you want the system to periodically judge whether the
current task is complete from camera images and robot state:

```yaml
env:
  OPENPI_TASK_OBSERVER_ENABLE: true
  OPENPI_TASK_OBSERVER_PYTHON: /path/to/observer/python
  OPENPI_TASK_OBSERVER_MODEL: /path/to/gemma/model
  OPENPI_TASK_OBSERVER_SPEC_FILE: task/prompt.txt
```

Observer completion rules live in `task/prompt.txt`.

If the observer is disabled or misconfigured, keep it disabled and run the main
robot workflow first.

## Policy Modes

Remote policy:

```yaml
env:
  OPENPI_POLICY_BACKEND: auto
  OPENPI_K8S_NAMESPACE: ...
  OPENPI_POLICY_LOCAL_PORT: 8000
  OPENPI_POLICY_REMOTE_PORT: 8000
```

Local PyTorch policy:

```yaml
env:
  OPENPI_POLICY_BACKEND: pytorch
  OPENPI_PYTORCH_RUNTIME_PYTHON: /path/to/torch/python
  OPENPI_PYTORCH_CHECKPOINT_DIR: /path/to/pytorch/checkpoint
  OPENPI_PYTORCH_DEVICE: cuda
```

## Notes

- `support/kubeconfig.yaml` is treated as local sensitive config and should not
  be committed.
- `docs/` and `tests/` are local-only by default.
- Use `tools/doctor.py --section runtime` or `tools/doctor.py --section observer`
  for narrower checks.
