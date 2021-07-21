docker run \
            --gpus all -it \
            --rm \
            --ipc=host \
            -v /media/alexey/HDD5/Mapillary:/workspace/Mapillary \
            -v /home/alexey/development/Resnet_OC:/workspace/Resnet_OC \
            -v /home/alexey/development/best_models/:/workspace/best_models \
            resnet_oc
