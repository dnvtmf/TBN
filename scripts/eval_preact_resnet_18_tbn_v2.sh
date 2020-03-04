#!/usr/bin/env bash
cd "$(dirname $0)/.."

python3 ./imagenet.py --load pretrained/preact_resnet_18_TBNv2.pth -e -a preact_resnet_18 ~/data/ImageNet --gpu 0
