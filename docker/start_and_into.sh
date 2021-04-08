docker run \
       	--gpus all -it \
       	--rm \
		--ipc=host \
       	-v /home/alexey/development/Resnet_OC:/home/Resnet_OC \
	-v /media/alexey/HDD/Mapillary:/home/Mapillary \
	atorch
