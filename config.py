import argparse
import sys
import os
from time import gmtime, strftime

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--expt_name', type=str)
    ### PATHS ###
    parser.add_argument('--pretrained_model_path', type=str, help="Datasets are stored in this directory", default="/mnt/store/knaraya4/hyp-oc/pretrained_weights/vgg_face_dag.pth")
    parser.add_argument('--data_root', type=str, help="Datasets are stored in this directory", default="/mnt/store/knaraya4/data")
    parser.add_argument('--csv_root', type=str, help="Data csv files are stored here", default="/mnt/store/knaraya4/hyp-oc/data")
    parser.add_argument('--save_root', type=str, help="Weights are saved here", default="/mnt/store/knaraya4/hyp-oc/weights")
    parser.add_argument('--log_root', type=str, help="Training logs are saved here", default="/mnt/store/knaraya4/hyp-oc/results")
    ### Training ###
    parser.add_argument('--dataset', type=str, help="ROSEYoutu" or "ReplayAttack" or "OULU_NPU" or "CASIA_MFSD" or "MSU_MFSD" or "OCI" or "OMI" or "OCM" or "ICM")
    parser.add_argument('--device', type=str, default="0", help="0" or "1" or "2")
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch_size_train', type=int, default=8)
    parser.add_argument('--batch_size_val', type=int, default=128)
    parser.add_argument('--val_check_after_epoch', type=int, default=1)
    parser.add_argument('--save_for_each_val_epoch', type=bool, default=False)
    parser.add_argument('--optim_lr', type=float, default=1e-6)
    parser.add_argument('--optim_weight_decay', type=float, default=1e-6)
    parser.add_argument('--std_dev', type=float, default=1)
    parser.add_argument('--feature_dimension', type=int, default=4096)
    parser.add_argument('--alpha', type=float, default=0.8)
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--curvature', type=float, default=0.1, help="Curvature of the hyperbolic ball")
    parser.add_argument('--finetune_params', type=str, default='conv5_1.weight,conv5_1.bias,conv5_2.weight,conv5_2.bias,conv5_3.weight,conv5_3.bias,fc6.weight,fc6.bias')

    args = parser.parse_args()
    args.fintune_params = [str(x) for x in args.finetune_params.split(',')]
    
    return args
