#!/bin/sh
python3 val.py --data-dir /workspace/Mapillary --model resnet_ocr --height 1080 --batch-size 1 --wandb --model-path ./save/resnet_ocr/model-3.pth --project-name OC