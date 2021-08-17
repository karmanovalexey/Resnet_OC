docker run \
            --gpus all -it \
            --rm \
            --ipc=host \
            -v /home/adeshkin/projects/tools/Mapillary/:/workspace/Mapillary \
            -v /home/adeshkin/projects/p_seg_dyn_map/kitti_360_track_0010_image_00:/workspace/Kitti \
            -v /home/karmanov_aa/Resnet_OC:/workspace/Resnet_OC \
            -v /home/karmanov_aa/best_models:/workspace/best_models \
            -v /datasets/KITTI-360:/datasets/KITTI-360 \
            resnet_oc
