docker run \
            --gpus all -it \
            --rm \
            --ipc=host \
            -v /media/alexey/HDD5/Mapillary:/workspace/Mapillary_2 \
            -v /home/adeshkin/projects/tools/Mapillary:/workspace/Mapillary \
            -v /home/karmanov_aa/Resnet_OC:/workspace/Resnet_OC \
            -v /home/karmanov_aa/taganrog:/workspace/video \
            resnet_oc
