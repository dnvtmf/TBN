# TBN
This is the implementation of TBN/TBNv2.

## Requirements
- Install PyTorch and torchvision
- Download the ImageNet dataset from http://www.image-net.org/
    - Then, and move validation images to labeled subfolders, using [the following shell script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)

## Results
| Arch | top-1 accuracy |  top-5 accuracy |
| ---- | ----- | ------ |
| AlexNet (full) | 61.6 | 82.9 |
| AlexNet (TBNv2) | 54.9 | 77.8 |
| PreActResNet18 (full) | 70.3 | 89.3 |
|  PreActResNet18 (TBNv2) | 59.7 | 82.1 |
| PreActResNet34 (full) | 73.3 | 76.4 |
|  PreActResNet34 (TBNv2) | 63.4 | 849 |
| PreActResNet50 (full) | 76.4 | 93.2 |
|  PreActResNet50 (TBNv2) | 66.6 | 86.7 |

## Evaluate

1. Download pretrained model
    - alexnet_TBNv2: [BaiduPan](https://pan.baidu.com/s/16bA11mfofDr0A6CglfLjBg) (password: 3rh2)
    - preact_resnet_18_TBNv2: Coming soon
    - preact_resnet_34_TBNv2: Coming soon
    - preact_resnet_50_TBNv2: [BaiduPan](https://pan.baidu.com/s/1oQz7u3hQkGyIhNs-DsFueQ ) (password: cmq7)
2. run command (see scripts/*)
```bash
python3 ./imagenet.py --load pretrained/alexnet_TBNv2.pth -e -a alexnet --gpu 0 ~/data/ImageNet
```

## Train

Please refer [pytorch/examples/imagenet](https://github.com/pytorch/examples/tree/master/imagenet)

Use `--ternary-delta=0.5`, `--ternary-order=2`, `--ternary-momentum=0.1` and `--ternary-no-scale` to set the
 hyper-parameter of TBN/TBNv2.
- For TBN, you need `--ternary-no-scale` option.
- when `--ternary-momentum` <= 0, the threshold value of ternary is fixed as `--ternary-delta` rather than calculating
 based on inputs.

## Citation

    @InProceedings{Wan_2018_ECCV,
    author = {Wan, Diwen and Shen, Fumin and Liu, Li and Zhu, Fan and Qin, Jie and Shao, Ling and Tao Shen, Heng},
    title = {TBN: Convolutional Neural Network with Ternary Inputs and Binary Weights},
    booktitle = {The European Conference on Computer Vision (ECCV)},
    month = {September},
    year = {2018}
    }