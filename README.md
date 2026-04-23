# OpenPI-AUBO i5 Scripts

杩欎釜浠撳簱鍙寘鍚満鍣ㄤ汉渚ц剼鏈紝璐熻矗锛?

- 鍦ㄧ嚎鎺ㄧ悊鎵ц
- 鐪熸満鏁版嵁閲囬泦
- 瑙傛祴鏋勫缓
- AUBO 鎺у埗杈呭姪閫昏緫
- 浠诲姟瀹屾垚鍒ゆ柇

榛樿鐩綍褰㈡€侊細

```text
openpi/
鈹溾攢 repo/        # OpenPI 涓讳粨搴?
鈹溾攢 OpenPi-Auboi5/
鈹溾攢 aubo_sdk/
鈹斺攢 captures/
```

## 浠撳簱缁撴瀯

```text
OpenPi-Auboi5/
鈹溾攢 config       # 鏍圭洰褰曞崟鏂囦欢閰嶇疆
鈹溾攢 tools/
鈹溾攢 utils/
鈹溾攢 support/
鈹溾攢 task/
鈹溾攢 data/
鈹溾攢 main.py
鈹斺攢 collect_data.py
```

`utils/` 鏀剧幆澧冦€佽矾寰勫拰閫氱敤宸ュ叿銆?
`support/` 鏀炬満鍣ㄤ汉鎺у埗銆佺瓥鐣ュ姞杞姐€佽娴嬨€乀UI 鍜?observer 鐩稿叧閫昏緫銆?

## 蹇€熷紑濮?

1. 鐩存帴缂栬緫鏍圭洰褰?`config`

鑷冲皯纭杩欎簺瀛楁锛?

- `OPENPI_RUNTIME_PYTHON`
- `OPENPI_SDK_ROOT`
- `OPENPI_ROBOT_IP`
- `OPENPI_CHECKPOINT_DIR`
- `OPENPI_TASK_OBSERVER_PYTHON`
- `OPENPI_TASK_OBSERVER_MODEL`
- 涓插彛鐩稿叧閰嶇疆

2. 杩愯鐜妫€鏌ワ細

```bash
python3 tools/doctor.py
```

鍙鏌ヤ富杩愯鐜锛?

```bash
python3 tools/doctor.py --section runtime
```

鍙鏌?observer 鐜锛?

```bash
python3 tools/doctor.py --section observer
```

3. 鍚姩鍦ㄧ嚎鎺ㄧ悊锛?

```bash
python3 main.py
```

4. 鍚姩鏁版嵁閲囬泦锛?

```bash
python3 collect_data.py
```

## 閰嶇疆鍔犺浇瑙勫垯

榛樿浼氭寜涓嬮潰椤哄簭鍔犺浇閰嶇疆锛?

1. `OPENPI_ENV_FILE` 鎸囧悜鐨勬樉寮忔枃浠?
2. 浠撳簱鏍圭洰褰曚笅鐨?`config`

鐜板湪涓嶅啀浣跨敤 `config/` 鐩綍瀛樻斁 `.env` 妯℃澘銆?

## 鐜渚濊禆

杩欎釜浠撳簱涓嶆槸瀹屾暣鍗曚粨鐜锛岃繍琛屽墠浠嶉渶瑕佸噯澶囧閮ㄤ緷璧栥€?

### 鐩綍渚濊禆

榛樿瑕佹眰浠ヤ笅鐩綍涓庡綋鍓嶄粨搴撳苟鍒楀瓨鍦細

- `../repo`
- `../aubo_sdk`

### Python 鐜

鎺ㄨ崘涓ゅ Python锛?

- OpenPI 涓荤幆澧?
  鐢?`OPENPI_RUNTIME_PYTHON` 鎸囧悜
- Observer / Gemma 鐜
  鐢?`OPENPI_TASK_OBSERVER_PYTHON` 鎸囧悜

鏈湴 PyTorch worker 涔熷彲浠ュ崟鐙敤涓€濂楃幆澧冿紝鐢?`OPENPI_PYTORCH_RUNTIME_PYTHON` 鎸囧悜銆?

### 绯荤粺鍜岃澶囦緷璧?

杩愯鍓嶉€氬父杩橀渶瑕侊細

- Jetson + JetPack / CUDA
- AUBO SDK
- Orbbec Python SDK
- 涓插彛璁惧
- Gemma 妯″瀷鐩綍

`tools/doctor.py` 浼氬敖閲忔彁鍓嶆毚闇茬己椤癸紝浣嗕笉鏇夸唬绯荤粺灞傚畨瑁呫€?

## 涓昏鍏ュ彛

- `main.py`: 鍦ㄧ嚎鎺ㄧ悊鎵ц鍏ュ彛
- `collect_data.py`: 鐪熸満鏁版嵁閲囬泦鍏ュ彛
- `support/load_policy.py`: 鏈湴 / 杩滅绛栫暐缁熶竴鍔犺浇
- `support/get_obs.py`: 瑙傛祴鏋勫缓
- `support/task_observer.py`: 浠诲姟瀹屾垚鍒ゆ柇

## PyTorch Local Backend

鏈湴 PyTorch 鎺ㄧ悊褰撳墠閫氳繃鐙珛 worker 杩涚▼杩愯锛?

- 涓昏繘绋嬬户缁礋璐ｇ浉鏈恒€佹満鍣ㄤ汉鍜屼富鎺у埗閫昏緫
- PyTorch policy worker 杩愯鍦?`OPENPI_PYTORCH_RUNTIME_PYTHON` 鎸囧悜鐨勭幆澧?
- worker 鐜闇€瑕佽兘鍔犺浇 CUDA Torch 鍜岃浆鎹㈠悗鐨?`model.safetensors`

甯哥敤鍙橀噺锛?

- `OPENPI_POLICY_BACKEND=pytorch`
- `OPENPI_PYTORCH_CHECKPOINT_DIR=...`
- `OPENPI_PYTORCH_RUNTIME_PYTHON=/home/niic/openpi/miniforge3/envs/openpi-py310-torch/bin/python`
- `OPENPI_PYTORCH_DEVICE=cuda`
- `OPENPI_SAMPLE_NUM_STEPS=5`

## 鍏煎璇存槑

- `support/kubeconfig.yaml` 鎸夋晱鎰熸枃浠跺鐞嗭紝榛樿涓嶇撼鍏?git
