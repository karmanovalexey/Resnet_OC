#!/bin/sh
python3 train.py --data-dir /workspace/Mapillary --model resnet_oc --height 600 --num-epochs 12 --wandb --batch-size 1 --pretrained --project-name Resnet-MOC-Training --save-dir resnet_oc_0