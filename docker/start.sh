docker run \
            --gpus all -it \
            --rm \
            --ipc=host \
            -v /media/alexey/HDD5/Mapillary:/workspace/Mapillary \
            -v /home/alexey/development/Resnet_OC:/workspace/Resnet_OC \
            -v /home/karmanov_aa/taganrog:/workspace/video \
            resnet_oc
