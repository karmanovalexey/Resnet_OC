docker run \
            --gpus all -it \
            --rm \
            --ipc=host \
            -v /media/alexey/HDD2/Mapillary:/workspace/Mapillary \
            -v /home/alexey/development/Resnet_OC:/workspace/Resnet_OC \
            resnet_oc