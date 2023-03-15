# DiffRate
This a Pytorch implementation of our paper "DiffRate : Differentiable Compression Rate for Efficient Vision Transformers"


## Requirements
```
- python >= 3.8
- pytorch >= 1.12.1  # For scatter_reduce
- torchvision        # With matching version for your pytorch install
```


## Data Preparation
- The ImageNet dataset should be prepared as follows:
```
ImageNet
├── train
│   ├── folder 1 (class 1)
│   ├── folder 2 (class 2)
│   ├── ...
├── val
│   ├── folder 1 (class 1)
│   ├── folder 2 (class 2)
│   ├── ...

```

## Pre-Trained Models
Our proposed DiffRate is designed to operate utilizing the officially endorsed pre-trained models of MAE and DeiT. To facilitate seamless integration, our code is programmed to automatically download and load these pre-trained models. However, users who prefer manual downloads can acquire the pre-trained MAE models via this [link](https://github.com/facebookresearch/mae/blob/main/FINETUNE.md), and the pre-trained DeiT models through this [link](https://github.com/facebookresearch/deit/blob/main/README_deit.md).
 


## Evaluation
We provide the discovered compression rates in the `compression_rate.json` file. To evaluate these rates, utilize the `--eval_with_compression_rate` option, which will load the appropriate compression rate from `compression_rate.json` based on the specified `model` and `target_flops`.

- For the `ViT-S (DeiT)` model, we currently offer support for the `--target_flops` option with `{2.3,2.5,2.7,2.9,3.1}`. To illustrate, an example evaluating the `ViT-S (DeiT)` model with `2.9G` FLOPs would be:
```
python main.py --eval --eval_with_compression_rate --data-path $path_to_imagenet$ --model vit_deit_small_patch16_224 --target_flops 2.9
```
This should give:
```
Acc@1 79.538 Acc@5 94.828 loss 0.902 flops 2.905
```
- For the `ViT-B (DeiT)` model, we currently offer support for the `--target_flops` option with `{8.7,10.0,10.4,11.5,12.5}`. To illustrate, an example evaluating the `ViT-B (DeiT)` model with `11.5G` FLOPs would be:
```
python main.py --eval --eval_with_compression_rate --data-path $path_to_imagenet$ --model vit_deit_base_patch16_224 --target_flops 11.5
```
This should give:
```
Acc@1 81.498 Acc@5 95.404 loss 0.861 flops 11.517
```
- For the `ViT-B (MAE)` model, we currently offer support for the `--target_flops` option with `{8.7,10.0,10.4,11.5}`. To illustrate, an example evaluating the `ViT-B (MAE)` model with `11.5G` FLOPs would be:
```
python main.py --eval --eval_with_compression_rate --data-path $path_to_imagenet$ --model vit_base_patch16_mae --target_flops 11.5
```
This should give:
```
Acc@1 82.864 Acc@5 96.148 loss 0.794 flops 11.517
```
- For the `ViT-L (MAE)` model, we currently offer support for the `--target_flops` option with `{31.0,34.7,38.5,42.3,46.1}`. To illustrate, an example evaluating the `ViT-L (MAE)` model with `42.3G` FLOPs would be:
```
python main.py --eval --eval_with_compression_rate --data-path $path_to_imagenet$ --model vit_large_patch16_mae --target_flops 42.3
```
This should give:
```
Acc@1 85.658 Acc@5 97.442 loss 0.683 flops 42.290
```
- For the `ViT-H (MAE)` model, we currently offer support for the `--target_flops` option with `{83.7,93.2,103.4,124.5}`. To illustrate, an example evaluating the `ViT-H (MAE)` model with `103.4G` FLOPs would be:
```
python main.py --eval --eval_with_compression_rate --data-path $path_to_imagenet$ --model vit_huge_patch14_mae --target_flops 103.4
```
This should give:
```
Acc@1 86.664 Acc@5 97.894 loss 0.602 flops 103.337
```


## Training

To find the optimal compression rate by proposed `DiffRate`, run the following code:
```
python -m torch.distributed.launch \
--nproc_per_node=4 --use_env  \
--master_port 29513 main.py \
--arch-lr 0.01 --arch-min-lr 0.001 \
--epoch 3 --batch-size 256 \
--data-path $path_to_imagenet$ \
--output_dir $path_to_save_log$ \
--model $model_name$ \
--target_flops $target_flops$
```
- supported `$model_name$`: `{vit_deit_tiny_patch16_224,vit_deit_small_patch16_224,vit_deit_base_patch16_224,vit_base_patch16_mae,vit_large_patch16_mae,vit_huge_patch14_mae}`
- supported `$target_flops$`: a floating point number

For example, search a `2.9G` compression rate schedule for `ViT-S (DeiT)`:
```
python -m torch.distributed.launch \
--nproc_per_node=4 --use_env  \
--master_port 29513 main.py \
--arch-lr 0.01 --arch-min-lr 0.001 \
--epoch 3 --batch-size 256 \
--data-path $path_to_imagenet$ \
--output_dir $path_to_save_log$ \
--model vit_deit_small_patch16_224 \
--target_flops 2.9
```