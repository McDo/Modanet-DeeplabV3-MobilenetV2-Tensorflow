### NOTE
> i. For `mobilenetv2_dm05` model, change the `depth_multiplier` FLAG to `0.5` in `common.py`, else change it back to 1.0.

> ii. If custom dataset is used for training but want to reuse the pre-trained feature encoder, try adding
```
--initialize_last_layer=False
--last_layers_contain_logits_only=False
``` 

> iii. When fine_tune_batch_norm=True, use at least batch size larger than **12** (batch size more than **16** is better). Otherwise, one could use smaller batch size and set fine_tune_batch_norm=False.

> iv. When running `python train.py` in colab, using `!python` instead of `%%bash python`, otherwise the notebook wouldn't print anything out.

> v. We always set crop_size = output_stride * k + 1, where k is an integer. When working on PASCAL images, the largest dimension is 512. Thus, we set crop_size = 513 = 16 * 32 + 1 > 512. Similarly, we set eval_crop_size = 1025x2049 for Cityscapes images.

## deeplab modanet training
```
python train.py \
    --logtostderr \
    --training_number_of_steps=30000 \
    --train_split="train" \
    --model_variant="mobilenet_v2" \
    --train_crop_size="513,513" \
    --train_batch_size=8 \
    --dataset="modanet_seg" \
    --fine_tune_batch_norm=True \
    --tf_initial_checkpoint=./deeplabv3_mnv2_pascal_trainval_2018_01_29/model.ckpt \
    --train_logdir=./train_logdir \
    --dataset_dir=./tfrecord \
    --initialize_last_layer=False \
    --last_layers_contain_logits_only=False
```