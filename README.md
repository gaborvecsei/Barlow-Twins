# Barlow Twins

Unofficial implementation of the [Barlow Twins Self-Supervised Learning method](https://arxiv.org/abs/2103.03230)

# Setup

## Docker

```bash
docker build -t barlow .
```

```bash
docker run --rm \
           -t \
           -u $(id -u):$(id -g) \
           --gpus all \
           -v $(pwd):/code \
           -v <DATASET_FOLDER_PATH>:/data \
           -w /code \
           barlow \
           python train.py --name my_test /data
```

**"Interactive mode"**

```bash
docker run --rm \
           -it \
           -u $(id -u):$(id -g) \
           --gpus all \
           -v $(pwd):/code \
           -v <DATASET_FOLDER_PATH>:/data \
           -w /code \
           barlow

# Run this in the container
CUDA_VISIBLE_DEVICES=0 python train.py --name my_test /data
```

# TODOs

- Test multiple optimization methods (LARS is what was used in the paper)
- Choose or use custom backbone
- Save model
    - Save frequency
    - Save only when loss improved
