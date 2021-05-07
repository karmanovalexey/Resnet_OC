#!/bin/sh
python3 train.py --data-dir /workspace/Mapillary --model resnet_moc --height 600 --num-epochs 30 --batch-size 6 --wandb --pretrained --project-name Resnet-MOC-Training --save-dir resnet_moc_1
