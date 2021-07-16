#!/bin/sh
python3 train.py --data-dir /workspace/Mapillary  --model resnet_funmoc --loss Focal --height 1080 --wandb  --num-epochs 100 --batch-size 3 --project-name MOC --save-dir MOC
