python train.py \
    --expt_name roseyoutu \
    --dataset ROSEYoutu \
    --device 0 \
    --epochs 60 \
    --batch_size_train 8 \
    --batch_size_val 128 \
    --val_check_after_epoch 1 \
    --save_for_each_val_epoch True \
    --optim_lr 1e-6 \
    --optim_weight_decay 1e-6 \
    --std_dev 1 \
    --feature_dimension 4096 \
    --alpha 0.8 \
    --curvature 0.1 \
    --resume True

python train.py \
    --expt_name replayattack \
    --dataset ReplayAttack \
    --device 0 \
    --epochs 30 \
    --batch_size_train 32 \
    --batch_size_val 128 \
    --val_check_after_epoch 1 \
    --save_for_each_val_epoch True \
    --optim_lr 1e-6 \
    --optim_weight_decay 1e-6 \
    --std_dev 1 \
    --feature_dimension 4096 \
    --alpha 0.8 \
    --curvature 0.1 

python train.py \
    --expt_name oulu_npu \
    --dataset OULU_NPU \
    --device 0 \
    --epochs 60 \
    --batch_size_train 32 \
    --batch_size_val 64 \
    --val_check_after_epoch 1 \
    --save_for_each_val_epoch True \
    --optim_lr 1e-6 \
    --optim_weight_decay 1e-6 \
    --std_dev 1 \
    --feature_dimension 4096 \
    --alpha 0.8 \
    --curvature 0.1 

python train.py \
    --expt_name casia_mfsd \
    --dataset CASIA_MFSD \
    --device 0 \
    --epochs 300 \
    --batch_size_train 8 \
    --batch_size_val 32 \
    --val_check_after_epoch 1 \
    --save_for_each_val_epoch True \
    --optim_lr 1e-6 \
    --optim_weight_decay 1e-6 \
    --std_dev 1 \
    --feature_dimension 4096 \
    --alpha 0.8 \
    --curvature 0.1 

python train.py \
    --expt_name msu_mfsd \
    --dataset MSU_MFSD \
    --device 0 \
    --epochs 300 \
    --batch_size_train 8 \
    --batch_size_val 32 \
    --val_check_after_epoch 1 \
    --save_for_each_val_epoch True \
    --optim_lr 1e-6 \
    --optim_weight_decay 1e-6 \
    --std_dev 1 \
    --feature_dimension 4096 \
    --alpha 0.8 \
    --curvature 0.1 

python train.py \
    --expt_name oci \
    --dataset OCI \
    --device 0 \
    --epochs 60 \
    --batch_size_train 32 \
    --batch_size_val 32 \
    --val_check_after_epoch 1 \
    --save_for_each_val_epoch True \
    --optim_lr 1e-6 \
    --optim_weight_decay 1e-6 \
    --std_dev 1 \
    --feature_dimension 4096 \
    --alpha 0.8 \
    --curvature 0.1 

python train.py \
    --expt_name omi \
    --dataset OMI \
    --device 0 \
    --epochs 60 \
    --batch_size_train 32 \
    --batch_size_val 32 \
    --val_check_after_epoch 1 \
    --save_for_each_val_epoch True \
    --optim_lr 1e-6 \
    --optim_weight_decay 1e-6 \
    --std_dev 1 \
    --feature_dimension 4096 \
    --alpha 0.8 \
    --curvature 0.1 

python train.py \
    --expt_name ocm \
    --dataset OCM \
    --device 0 \
    --epochs 60 \
    --batch_size_train 32 \
    --batch_size_val 64 \
    --val_check_after_epoch 1 \
    --save_for_each_val_epoch True \
    --optim_lr 1e-6 \
    --optim_weight_decay 1e-6 \
    --std_dev 1 \
    --feature_dimension 4096 \
    --alpha 0.8 \
    --curvature 0.1 

python train.py \
    --expt_name icm \
    --dataset ICM \
    --device 0 \
    --epochs 100 \
    --batch_size_train 32 \
    --batch_size_val 64 \
    --val_check_after_epoch 1 \
    --save_for_each_val_epoch True \
    --optim_lr 1e-6 \
    --optim_weight_decay 1e-6 \
    --std_dev 1 \
    --feature_dimension 4096 \
    --alpha 0.8 \
    --curvature 0.1 
