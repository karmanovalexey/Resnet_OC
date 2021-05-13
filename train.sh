#!/bin/sh
python3 train.py --data-dir /workspace/Mapillary --model resnet_ocr --height 600 --num-epochs 200 --resume --batch-size 5 --wandb --pretrained --project-name Resnet-MOC-Training --save-dir resnet_ocr_0
