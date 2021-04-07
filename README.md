# Mapillary_semantics
This repository uses deeplabv3+, unet and erfnet to train on mapillary. It uses weights &amp; biases


python val.py --model deeplab --model-dir deeplab_main_run --height 1080 --project-name "Evaluation" --data-dir /media/alexey/HDD/Mapillary --batch-size 1


python train.py --data-dir /home/Mapillary/ --model resnet_oc --height 1080 --num-epochs 1 --batch-size 1 --epochs-save 1 --save-dir resnet_oc
