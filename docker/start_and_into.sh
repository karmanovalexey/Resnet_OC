docker run \
       	--gpus all -it \
       	--rm \
	    --ipc=host \
       	-v /home/yudin/alexey/Resnet_OC:/home/Resnet_OC \
	-v /home/yudin/Datasets/Mapillary:/home/Mapillary \
	alexey-pytorch
