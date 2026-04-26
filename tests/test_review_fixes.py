from __future__ import annotations

import importlib
import subprocess
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


class CameraPairReadinessTest(unittest.TestCase):
    def _import_get_obs_with_stubs(self):
        fake_cv2 = types.ModuleType("cv2")
        fake_cv2.COLOR_BGR2RGB = 0
        fake_cv2.cvtColor = lambda image, _code: image
        sys.modules.setdefault("cv2", fake_cv2)

        fake_pyorbbec = types.ModuleType("pyorbbecsdk")
        fake_pyorbbec.Context = object
        fake_pyorbbec.Pipeline = object
        fake_pyorbbec.Config = object
        fake_pyorbbec.Device = object
        fake_pyorbbec.FormatConvertFilter = object
        fake_pyorbbec.VideoFrame = object
        fake_pyorbbec.OBFormat = types.SimpleNamespace(RGB=object(), BGR=object())
        fake_pyorbbec.OBConvertFormat = types.SimpleNamespace(RGB_TO_BGR=object())
        fake_pyorbbec.OBSensorType = types.SimpleNamespace(COLOR_SENSOR=object())
        fake_pyorbbec.OBPropertyID = types.SimpleNamespace(
            OB_PROP_COLOR_AUTO_EXPOSURE_BOOL=object(),
            OB_PROP_COLOR_AE_MAX_EXPOSURE_INT=object(),
            OB_PROP_COLOR_BRIGHTNESS_INT=object(),
            OB_PROP_COLOR_BACKLIGHT_COMPENSATION_INT=object(),
            OB_PROP_COLOR_EXPOSURE_INT=object(),
            OB_PROP_COLOR_GAIN_INT=object(),
        )
        sys.modules.setdefault("pyorbbecsdk", fake_pyorbbec)

        fake_image_tools = types.ModuleType("openpi_client.image_tools")
        fake_image_tools.resize_with_pad = lambda image, _h, _w: image
        fake_image_tools.convert_to_uint8 = lambda image: image
        fake_openpi_client = types.ModuleType("openpi_client")
        fake_openpi_client.image_tools = fake_image_tools
        sys.modules.setdefault("openpi_client", fake_openpi_client)
        sys.modules.setdefault("openpi_client.image_tools", fake_image_tools)

        return importlib.import_module("support.get_obs")

    def test_wait_until_ready_raises_when_frames_never_arrive(self):
        get_obs = self._import_get_obs_with_stubs()
        pair = object.__new__(get_obs.CameraPair)
        pair._lock = mock.MagicMock()
        pair._lock.__enter__.return_value = None
        pair._lock.__exit__.return_value = None
        pair._latest_frames = {"main": None, "wrist": None}

        with self.assertRaisesRegex(RuntimeError, "camera frames not ready"):
            pair._wait_until_ready(timeout_s=0.0)


class RemotePolicyRunnerPipeTest(unittest.TestCase):
    def test_port_forward_does_not_leave_stdout_stderr_as_pipes(self):
        fake_pytorch_support = types.ModuleType("support.pytorch_support")
        fake_pytorch_support.decode_worker_value = lambda value: value
        fake_pytorch_support.encode_worker_value = lambda value: value
        sys.modules.setdefault("support.pytorch_support", fake_pytorch_support)

        from support.load_policy import RemotePolicyRunner

        runner = RemotePolicyRunner(kubeconfig=Path("kubeconfig"), namespace="ns")
        with (
            mock.patch.object(Path, "exists", return_value=True),
            mock.patch.object(runner, "_find_pod", return_value="pod-1"),
            mock.patch("support.load_policy.time.sleep", return_value=None),
            mock.patch("support.load_policy.subprocess.Popen") as popen,
        ):
            proc = mock.MagicMock()
            proc.poll.return_value = None
            popen.return_value = proc

            runner._start_port_forward()

        _args, kwargs = popen.call_args
        self.assertIsNot(kwargs.get("stdout"), subprocess.PIPE)
        self.assertIsNot(kwargs.get("stderr"), subprocess.PIPE)


class CheckDataPassMarkerTest(unittest.TestCase):
    def test_pass_marker_is_ignored_when_checker_version_mismatches(self):
        import data.check_data as check_data

        metadata = {
            check_data.CHECK_PASSED_KEY: True,
            check_data.CHECKER_VERSION_KEY: check_data.CHECKER_VERSION - 1,
        }

        with (
            mock.patch.object(Path, "exists", return_value=True),
            mock.patch.object(check_data, "read_episode_metadata", return_value=metadata),
            mock.patch("data.check_data.np.load", side_effect=AssertionError("recheck attempted")),
        ):
            with self.assertRaisesRegex(AssertionError, "recheck attempted"):
                check_data.inspect_episode(Path("episode_0000"), raw_capture_fps=30.0)


class YawOnlyStateModeTest(unittest.TestCase):
    def _import_convert_data_with_stubs(self):
        fake_pa = types.ModuleType("pyarrow")
        fake_pq = types.ModuleType("pyarrow.parquet")
        fake_imageio = types.ModuleType("imageio")
        fake_imageio_v2 = types.ModuleType("imageio.v2")
        fake_tyro = types.ModuleType("tyro")
        fake_tyro.cli = lambda _fn: None
        sys.modules.setdefault("pyarrow", fake_pa)
        sys.modules.setdefault("pyarrow.parquet", fake_pq)
        sys.modules.setdefault("imageio", fake_imageio)
        sys.modules.setdefault("imageio.v2", fake_imageio_v2)
        sys.modules.setdefault("tyro", fake_tyro)
        return importlib.import_module("data.convert_data")

    def test_convert_rejects_legacy_j6_state_mode(self):
        convert_data = self._import_convert_data_with_stubs()
        states = __import__("numpy").zeros((2, 8), dtype=__import__("numpy").float32)

        with self.assertRaisesRegex(ValueError, "only yaw"):
            convert_data.infer_episode_state_format(
                Path("episode_0000"),
                states,
                {"state_mode": "j6"},
            )

    def test_check_data_rejects_legacy_j6_state_mode(self):
        import numpy as np
        import data.check_data as check_data

        class FakeImages(dict):
            pass

        arrays = {
            "states.npy": np.zeros((2, 8), dtype=np.float32),
            "actions.npy": np.zeros((2, 7), dtype=np.float32),
            "timestamps.npy": np.array([0.0, 0.02], dtype=np.float32),
            "env_steps.npy": np.array([0, 1], dtype=np.int64),
            "images.npz": FakeImages(
                main_images=np.zeros((2, 224, 224, 3), dtype=np.uint8),
                wrist_images=np.zeros((2, 224, 224, 3), dtype=np.uint8),
            ),
        }

        def fake_load(path, *args, **kwargs):
            return arrays[Path(path).name]

        with (
            mock.patch.object(Path, "exists", return_value=True),
            mock.patch.object(
                check_data,
                "read_episode_metadata",
                return_value={"state_mode": "j6", "state_dim": 8, "action_dim": 7, "fps": 50, "n_frames": 2},
            ),
            mock.patch("data.check_data.np.load", side_effect=fake_load),
        ):
            with self.assertRaisesRegex(ValueError, "only yaw"):
                check_data.inspect_episode(Path("episode_0000"), raw_capture_fps=30.0)


if __name__ == "__main__":
    unittest.main()
