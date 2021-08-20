docker run \
            --gpus all -it \
            --rm \
            --ipc=host \
            --net=host \
            -v /media/alexey/Data/Datasets/Mapillary:/workspace/Mapillary \
            -v /home/alexey/development/Resnet_OC:/workspace/Resnet_OC \
            -v /home/alexey/development/model_weights:/workspace/model_weights \
            resnet_oc
