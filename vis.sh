#!/bin/sh
python3 vis.py --data-dir /workspace/Mapillary/val/1920_1080/images/ --dataset Mapillary --model resnet_ocold --height 1080 --load-dir /workspace/best_models/resnet_oc_best.pth --save-dir ./visual/col_map/ocold