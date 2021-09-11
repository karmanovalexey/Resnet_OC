docker run \
            --gpus all -it \
            --rm \
            --ipc=host \
            --net=host \
            -v /datasets/KITTI-360/:/workspace/KITTI-360 \
            -v /datasets/Mapillary/mapillary-vistas-dataset_public_v1.1:/workspace/Mapillary \
            -v /home/karmanov_aa/Resnet_OC:/workspace/Resnet_OC \
            -v /home/karmanov_aa/semantic-segmentation:/workspace/ss \
            resnet_oc
