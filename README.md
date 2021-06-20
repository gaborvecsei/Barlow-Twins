# Barlow Twins

Unofficial implementation of the [Barlow Twins Self-Supervised Learning method](https://arxiv.org/abs/2103.03230)

`docker build -t barlow .`

`docker run --rm -it -u $(id -u):$(id -g) --gpus all -v $(pwd):/code -v DATASET:/data -w /code barlow`

# TODOs

- Learning rate scheduler
- Test multiple optimization methods (LARS is what was used in the paper)
- Choose or use custom backbone
- Compact training script
- Comments for functions
- Augmentation parameters tuned
- New augmentations added (if needed)
- Separate logging of the two parts of the loss
- Save model
    - Save frequency
    - Save only when loss improved
