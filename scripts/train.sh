#!/bin/sh
python3 train.py --data-dir /workspace/Mapillary  --model resnet_moc --loss Focal --height 1080 --wandb  --num-epochs 100 --batch-size 6 --pretrained --project-name MOC --save-dir MOC
