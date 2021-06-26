docker run \
            --gpus all -it \
            --rm \
            --ipc=host \
            -v /home/raaicv/karmanov/Mapillary:/workspace/Mapillary \
            -v /home/raaicv/karmanov/Resnet_OC:/workspace/Resnet_OC \
            resnet_oc
