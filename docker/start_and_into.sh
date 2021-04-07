docker run \
       	--gpus all -it \
       	--rm \
	--shm-size 8G \
       	-v /home/yudin/alexey/Resnet_OC:/home/Resnet_OC \
	-v /home/yudin/Datasets/Mapillary:/home/Mapillary \
	alexey-pytorch
