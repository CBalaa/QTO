# FB15K
CUDA_VISIBLE_DEVICES=0 python main.py  \
    --dataset FB15K --score_rel True --model ComplEx --rank 1000 \
    --learning_rate 0.1 --batch_size 100 --lmbda 0.01 --w_rel 0.1 --max_epochs 100

# FB15k-237
CUDA_VISIBLE_DEVICES=0 python main.py \
    --dataset FB15k-237 --score_rel True --model ComplEx --rank 1000 \
    --learning_rate 0.1 --batch_size 1000 --lmbda 0.05 --w_rel 4 --max_epochs 100

# NELL995
CUDA_VISIBLE_DEVICES=0 python main.py \
    --dataset NELL995 --score_rel True --model ComplEx --rank 1000 \
    --learning_rate 0.1 --batch_size 1000 --lmbda 0.05 --w_rel 0 --max_epochs 100