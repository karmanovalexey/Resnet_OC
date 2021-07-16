docker run \
            --gpus all -it \
            --rm \
            --ipc=host \
<<<<<<< HEAD
            -v /media/alexey/HDD5/Mapillary:/workspace/Mapillary_2 \
            -v /home/adeshkin/projects/tools/Mapillary:/workspace/Mapillary \
            -v /home/karmanov_aa/Resnet_OC:/workspace/Resnet_OC \
            -v /home/karmanov_aa/taganrog:/workspace/video \
=======
            -v /home/raaicv/karmanov/Mapillary:/workspace/Mapillary \
            -v /home/raaicv/karmanov/Resnet_OC:/workspace/Resnet_OC \
>>>>>>> b5cd21de395caa5691dcbf2fb5d8d099efb8b0a9
            resnet_oc
