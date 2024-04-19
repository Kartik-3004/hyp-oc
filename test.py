import os
import numpy as np
import cv2
import sys
from tqdm import tqdm
import argparse
from datetime import datetime
import time
import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.distributions
import config
from models import hyp_classifier, Vgg_face_dag, load_vgg_face
from utils import save_checkpoint
from dataloader import ROSEYoutu, ReplayAttack, OULU_NPU, CASIA_MFSD, MSU_MFSD
import statistics
from loss import TPC_loss_hyp

def test(args):
    #Params and Config
    name = "-".join(args.list)
    file = open(f"{args.log_root}/{args.source_dataset}/{name}_{args.target_dataset}_test.txt", "a")
    sys.stdout = file
    print("---"*30)
    for arg in vars(args):
        num_space = 25 - len(arg)
        print(arg + " " * num_space + str(getattr(args, arg)))
    print("---"*30)
    device = "cuda:" + args.device

    #Dataloaders
    if args.target_dataset == "ROSEYoutu":
        testset = ROSEYoutu(split="test", csv_root=args.csv_root, data_root=args.data_root)
        test_dataloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size_test, shuffle=False, num_workers=4)
    elif args.target_dataset == "ReplayAttack":
        testset = ReplayAttack(split="test", csv_root=args.csv_root, data_root=args.data_root)
        test_dataloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size_test, shuffle=False, num_workers=4)
    elif args.target_dataset == "OULU_NPU":
        testset = OULU_NPU(split="test", csv_root=args.csv_root, data_root=args.data_root)
        test_dataloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size_test, shuffle=False, num_workers=4)
    elif args.target_dataset == "CASIA_MFSD":
        testset = CASIA_MFSD(split="test", csv_root=args.csv_root, data_root=args.data_root)
        test_dataloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size_test, shuffle=False, num_workers=4)
    elif args.target_dataset == "MSU_MFSD":
        testset = MSU_MFSD(split="test", csv_root=args.csv_root, data_root=args.data_root)
        test_dataloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size_test, shuffle=False, num_workers=4)
    elif args.target_dataset == "ROSEYoutu":
        testset = ROSEYoutu(split="test", csv_root=args.csv_root, data_root=args.data_root)
        test_dataloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size_test, shuffle=False, num_workers=4)
    elif args.target_dataset == "OCI":
        testset = MSU_MFSD(split="test", csv_root=args.csv_root, data_root=args.data_root)
        test_dataloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size_test, shuffle=False, num_workers=4)
    elif args.target_dataset == "OMI":
        testset = CASIA_MFSD(split="test", csv_root=args.csv_root, data_root=args.data_root)
        test_dataloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size_test, shuffle=False, num_workers=4)
    elif args.target_dataset == "OCM":
        testset = ReplayAttack(split="test", csv_root=args.csv_root, data_root=args.data_root)
        test_dataloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size_test, shuffle=False, num_workers=4)
    elif args.target_dataset == "ICM":
        testset = OULU_NPU(split="test", csv_root=args.csv_root, data_root=args.data_root)
        test_dataloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size_test, shuffle=False, num_workers=4)

    #Metric Initialization
    best_APCER = 1.0
    best_NPCER = 1.0
    best_ACER = 1.0
    best_EER = 1.0
    best_HTER = 1.0
    best_roc_auc = 0.0
    best_threshold = 0.0
    best_accuracy_threshold = 0.0
    metrics = {"APCER": 0, "NPCER":0, "ACER": 0, "EER": 0, "HTER": 0, "ROC_AUC_Score": 0, "Threshold": 0, "Accuracy_threshold": 0} 

    #Model Initialization
    vgg_face = load_vgg_face(device=device, weights_path=f"{args.pretrained_model_path}", return_layer='fc6')
    model = hyp_classifier(c=args.curvature).to(device)
    APCER_lis = []
    NPCER_lis = []
    AUC_lis = []
    HTER_lis = []
    for expt in args.list:
        filename = f"{args.save_root}/{args.source_dataset}/{expt}/best_epoch.pth"
        checkpoint = torch.load(filename, map_location=device)
        vgg_face.load_state_dict(checkpoint["encoder_state_dict"])
        model.load_state_dict(checkpoint["classifier_state_dict"])

        with torch.no_grad():
            vgg_face.eval()
            model.eval()
            labels_list = []
            predictions_list = []
            test_pbar = tqdm(test_dataloader, leave=True)
            for batch in test_pbar:
                #Feature extraction from VGG
                images, labels = batch
                images = images.to(device)
                labels = labels.to(device)
                features = vgg_face(images)
                classifier_features, classifier_output = model(features)

                #Predictions and Labels
                predictions = F.softmax(classifier_output, dim=1).cpu().numpy()[:, 1]
                labels = labels.cpu().numpy()
                labels_list.extend(labels)
                predictions_list.extend(predictions)


            labels_list = np.array(labels_list)
            predictions_list = np.array(predictions_list)

            #Calculate Metrics
            APCER, NPCER, ACER, EER, HTER, roc_auc, threshold, accuracy_threshold = statistics.calculate_metrics(labels_list, predictions_list)
            metrics["APCER"], metrics["NPCER"], metrics["ACER"], metrics["EER"], metrics["HTER"], metrics["ROC_AUC_Score"], metrics["Threshold"], metrics["Accuracy_threshold"] = APCER, NPCER, ACER, EER, HTER, roc_auc, threshold, accuracy_threshold
            APCER_lis.append(metrics["APCER"]*100)
            NPCER_lis.append(metrics["NPCER"]*100)
            HTER_lis.append(metrics["HTER"]*100)
            AUC_lis.append(metrics["ROC_AUC_Score"])
            #Print results
            print(f"\n##### TEST SET RESULTS  {expt} #####")
            print("APCER: ", metrics["APCER"]*100)
            print("NPCER: ", metrics["NPCER"]*100) 
            print("ACER: ", metrics["ACER"]*100)
            print("HTER: ", metrics["HTER"]*100)
            print("ROC_AUC_Score: ", metrics["ROC_AUC_Score"])
            print(f"Accuracy@{metrics['Threshold']}: ", metrics["Accuracy_threshold"]*100.0)

    APCER_lis = np.array(APCER_lis)
    NPCER_lis = np.array(NPCER_lis)
    AUC_lis = np.array(AUC_lis)
    HTER_lis = np.array(HTER_lis)
    
    print("\n\n")
    print("### Top 3 ###")
    print("APCER: ", np.mean(APCER_lis[np.argsort(APCER_lis)[:3]]), np.std(APCER_lis[np.argsort(APCER_lis)[:3]]))
    print("NPCER: ", np.mean(NPCER_lis[np.argsort(NPCER_lis)[:3]]), np.std(NPCER_lis[np.argsort(NPCER_lis)[:3]]))
    print("HTER: ", np.mean(HTER_lis[np.argsort(HTER_lis)[:3]]), np.std(HTER_lis[np.argsort(HTER_lis)[:3]]))
    print("AUC: ", np.mean(AUC_lis[np.argsort(AUC_lis)[-3:]]), np.std(AUC_lis[np.argsort(AUC_lis)[-3:]]))
    
    print("\n")
    print("### Top 5 ###")
    print("APCER: ", np.mean(APCER_lis), np.std(APCER_lis))
    print("NPCER: ", np.mean(NPCER_lis), np.std(NPCER_lis))
    print("HTER: ", np.mean(HTER_lis), np.std(HTER_lis))
    print("AUC: ", np.mean(AUC_lis), np.std(AUC_lis))

        
    sys.stdout = sys.__stdout__
    file.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_model_path', type=str, help="Datasets are stored in this directory", default="/mnt/store/knaraya4/hyp-oc/pretrained_weights/vgg_face_dag.pth")
    parser.add_argument('--data_root', type=str, help="Datasets are stored in this directory", default="/mnt/store/knaraya4/data")
    parser.add_argument('--csv_root', type=str, help="Data csv files are stored here", default="/mnt/store/knaraya4/hyp-oc/data")
    parser.add_argument('--save_root', type=str, help="Weights are saved here", default="/mnt/store/knaraya4/hyp-oc/weights")
    parser.add_argument('--log_root', type=str, help="Training logs are saved here", default="/mnt/store/knaraya4/hyp-oc/results")
    parser.add_argument('--source_dataset', type=str, help="ROSEYoutu" or "ReplayAttack" or "OULU_NPU" or "CASIA_MFSD" or "MSU_MFSD" or "OCI" or "OMI" or "OCM" or "ICM")
    parser.add_argument('--target_dataset', type=str, help="ROSEYoutu" or "ReplayAttack" or "OULU_NPU" or "CASIA_MFSD" or "MSU_MFSD" or "OCI" or "OMI" or "OCM" or "ICM")
    parser.add_argument('--device', type=str, default="0", help="0" or "1" or "2")
    parser.add_argument('--batch_size_test', type=int, default=32)
    parser.add_argument('--curvature', type=float, default=0.1, help="Curvature of the hyperbolic ball")
    parser.add_argument('--list', type=str, default='run_1,run_2,run_3,run_4,run_5', help="The name of the experiments")
    args = parser.parse_args()
    args.list = [str(x) for x in args.list.split(',')]
    test(args)