#!/usr/bin/env bash
cd "$(dirname $0)/.."

python3 ./imagenet.py --load pretrained/alexnet_TBNv2.pth -e -a alexnet ~/data/ImageNet --gpu 0
