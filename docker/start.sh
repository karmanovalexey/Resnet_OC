docker run \
            --gpus all -it \
            --rm \
            --ipc=host \
            -v /datasets/Mapillary_fullhd:/workspace/Mapillary \
            -v /home/karmanov_aa/Resnet_OC:/workspace/Resnet_OC \
            -v /home/karmanov_aa/taganrog:/workspace/video \
            resnet_oc
