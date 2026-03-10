# HMFM

This repository contains the MapTR-based implementation of our TITS paper:

**HMFM: A Robust Hierarchical Multimodal Fusion Module to Enhance Map Learning With Additional Information**.

## About This Repository

- This codebase is built on top of MapTR.
- It provides our HMFM implementation in the MapTR framework.
- If you want the HDMapNet-based version, please visit:
  - https://github.com/xjtu-cs-gao/SatforHDMap

## Quick Run

### 1. Install dependencies

```bash
pip install -r requirement.txt
pip install -v -e ./mmdetection3d
```

### 2. Prepare dataset

Follow the dataset preparation guide:

- `docs/prepare_dataset.md`

### 3. Train

```bash
bash tools/dist_train.sh <CONFIG_FILE> <NUM_GPUS>
```

Example:

```bash
bash tools/dist_train.sh projects/configs/<your_config>.py 8
```

### 4. Evaluate

```bash
bash tools/dist_test.sh <CONFIG_FILE> <CHECKPOINT_FILE> <NUM_GPUS>
```

### 5. Visualization

See:

- `docs/visualization.md`

## Notes

- This repository focuses on the MapTR-based HMFM implementation.
- For more detailed installation, training, and evaluation settings, see:
  - `docs/install.md`
  - `docs/train_eval.md`
