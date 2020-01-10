## deeplab modanet evaluation
```
python eval.py \
    --logtostderr \
    --eval_split="val" \
    --model_variant="mobilenet_v2" \
    --eval_crop_size="601,401" \
    --dataset="modanet_seg" \
    --output_stride=8 \
    --checkpoint_dir=./deeplabv3_mnv2_pascal_trainval_2018_01_29/trained \
    --eval_logdir=./eval_logdir \
    --dataset_dir=/PATH/TO/TFRECORD_DIR
```