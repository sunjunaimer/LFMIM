CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 LFMIMv2.py \
	    --rank 0 \
        --world_size 4 \
        --init_method env:// 