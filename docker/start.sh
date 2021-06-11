docker run \
            --gpus all -it \
            --rm \
            --ipc=host \
            -v /media/alexey/HDD1/Mapillary:/workspace/Mapillary \
            -v /home/alexey/development_2/Resnet_OC:/workspace/Resnet_OC \
            -v /media/alexey/HDD1/taganrog:/workspace/video \
            resnet_oc
