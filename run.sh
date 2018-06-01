# provided fixed arc
#fixed_arc="0 3 0 0 0 5 0 2 0 5 1 2 1 0 0 2 0 3 1 2"
#fixed_arc="$fixed_arc 1 0 1 0 0 4 0 3 1 2 3 2 1 0 0 5 0 4 1 2"

# searched arc
fixed_arc="1 6 0 1 1 0 0 6 1 1 0 4 0 0 4 1 0 4 1 4"
fixed_arc="$fixed_arc 0 5 1 1 1 6 0 2 0 5 0 3 0 0 0 5 2 3 0 1"

CUDA_VISIBLE_DEVICES=0 python train.py --dataset cifar10 --model enas --epochs 360 --data_augmentation --cutout --length 16 --fixed_arc "$fixed_arc"
