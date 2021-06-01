#!/bin/sh
python3 val.py --data-dir /workspace/Mapillary --model resnet_ocold --height 1080 --batch-size 1 --wandb --model-path ./save/mapillary_resnet_M_base_oc_focal_2_best.pth --project-name OC