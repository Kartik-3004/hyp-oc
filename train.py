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


def train(args):
    #Params and Config
    if os.path.isdir(os.path.join(args.save_root, f"{args.dataset}", args.expt_name)) == False:
        os.makedirs(os.path.join(args.save_root, f"{args.dataset}", args.expt_name))
    if os.path.isdir(os.path.join(args.log_root, f"{args.dataset}")) == False:
        os.makedirs(os.path.join(args.log_root, f"{args.dataset}"))
    file = open(f"{args.log_root}/{args.dataset}/{args.expt_name}_train.txt", "a")
    sys.stdout = file
    print("---"*30)
    for arg in vars(args):
        num_space = 25 - len(arg)
        print(arg + " " * num_space + str(getattr(args, arg)))
    print("---"*30)
    device = "cuda:" + args.device

    #Dataloaders
    if args.dataset == "ROSEYoutu":
        trainset = ROSEYoutu(split="train", csv_root=args.csv_root, data_root=args.data_root)
        valset = ROSEYoutu(split="test", csv_root=args.csv_root, data_root=args.data_root)
        train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size_train, shuffle=True, num_workers=4)
        val_dataloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size_val, shuffle=False, num_workers=4)
    elif args.dataset == "ReplayAttack":
        trainset = ReplayAttack(split="train", csv_root=args.csv_root, data_root=args.data_root)
        valset = ReplayAttack(split="val", csv_root=args.csv_root, data_root=args.data_root)
        train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size_train, shuffle=True, num_workers=4)
        val_dataloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size_val, shuffle=False, num_workers=4)
    elif args.dataset == "OULU_NPU":
        trainset = OULU_NPU(split="train", csv_root=args.csv_root, data_root=args.data_root)
        valset = OULU_NPU(split="val", csv_root=args.csv_root, data_root=args.data_root)
        train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size_train, shuffle=True, num_workers=4)
        val_dataloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size_val, shuffle=False, num_workers=4)
    elif args.dataset == "CASIA_MFSD":
        trainset = CASIA_MFSD(split="train", csv_root=args.csv_root, data_root=args.data_root)
        valset = CASIA_MFSD(split="val", csv_root=args.csv_root, data_root=args.data_root)
        train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size_train, shuffle=True, num_workers=4)
        val_dataloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size_val, shuffle=False, num_workers=4)
    elif args.dataset == "MSU_MFSD":
        trainset = MSU_MFSD(split="train", csv_root=args.csv_root, data_root=args.data_root)
        valset = MSU_MFSD(split="val", csv_root=args.csv_root, data_root=args.data_root)
        train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size_train, shuffle=True, num_workers=4)
        val_dataloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size_val, shuffle=False, num_workers=4)
    elif args.dataset == "ROSEYoutu":
        trainset = ROSEYoutu(split="train", csv_root=args.csv_root, data_root=args.data_root)
        valset = ROSEYoutu(split="test", csv_root=args.csv_root, data_root=args.data_root)
        train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size_train, shuffle=True, num_workers=4)
        val_dataloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size_val, shuffle=False, num_workers=4)
    elif args.dataset == "OCI":
        trainset_1 = OULU_NPU(split="train", csv_root=args.csv_root, data_root=args.data_root)
        trainset_2 = CASIA_MFSD(split="train", csv_root=args.csv_root, data_root=args.data_root)
        trainset_3 = ReplayAttack(split="train", csv_root=args.csv_root, data_root=args.data_root)
        trainset = torch.utils.data.ConcatDataset([trainset_1, trainset_2, trainset_3])
        valset = MSU_MFSD(split="val", csv_root=args.csv_root, data_root=args.data_root)
        train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size_train, shuffle=True, num_workers=4)
        val_dataloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size_val, shuffle=False, num_workers=4)
    elif args.dataset == "OMI":
        trainset_1 = OULU_NPU(split="train", csv_root=args.csv_root, data_root=args.data_root)
        trainset_2 = MSU_MFSD(split="train", csv_root=args.csv_root, data_root=args.data_root)
        trainset_3 = ReplayAttack(split="train", csv_root=args.csv_root, data_root=args.data_root)
        trainset = torch.utils.data.ConcatDataset([trainset_1, trainset_2, trainset_3])
        valset = CASIA_MFSD(split="val", csv_root=args.csv_root, data_root=args.data_root)
        train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size_train, shuffle=True, num_workers=4)
        val_dataloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size_val, shuffle=False, num_workers=4)
    elif args.dataset == "OCM":
        trainset_1 = OULU_NPU(split="train", csv_root=args.csv_root, data_root=args.data_root)
        trainset_2 = CASIA_MFSD(split="train", csv_root=args.csv_root, data_root=args.data_root)
        trainset_3 = MSU_MFSD(split="train", csv_root=args.csv_root, data_root=args.data_root)
        trainset = torch.utils.data.ConcatDataset([trainset_1, trainset_2, trainset_3])
        valset = ReplayAttack(split="val", csv_root=args.csv_root, data_root=args.data_root)
        train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size_train, shuffle=True, num_workers=4)
        val_dataloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size_val, shuffle=False, num_workers=4)
    elif args.dataset == "ICM":
        trainset_1 = MSU_MFSD(split="train", csv_root=args.csv_root, data_root=args.data_root)
        trainset_2 = CASIA_MFSD(split="train", csv_root=args.csv_root, data_root=args.data_root)
        trainset_3 = ReplayAttack(split="train", csv_root=args.csv_root, data_root=args.data_root)
        trainset = torch.utils.data.ConcatDataset([trainset_1, trainset_2, trainset_3])
        valset = OULU_NPU(split="val", csv_root=args.csv_root, data_root=args.data_root)
        train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size_train, shuffle=True, num_workers=4)
        val_dataloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size_val, shuffle=False, num_workers=4)

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
    if args.resume:
        checkpoint_path = f'{args.save_root}/{args.dataset}/{args.expt_name}/best_epoch.pth'
        checkpoint = torch.load(checkpoint_path, map_location=device)
        vgg_face.load_state_dict(checkpoint['encoder_state_dict'])
        model.load_state_dict(checkpoint['classifier_state_dict'])
        metrics = checkpoint['metrics']
        start_epoch = checkpoint['epoch']
    for p in model.parameters():
        p.requires_grad = True
    for p in vgg_face.parameters():
        p.requires_grad = False
    print("Finetuning the below parameters --> ")
    for name, p in vgg_face.named_parameters():
        if name in args.finetune_params:
            print(name)
            p.requires_grad = True

    #Optimizers and Loss function
    optimizer_dict = [
        {"params": filter(lambda p: p.requires_grad, model.parameters()), "lr": args.optim_lr},
        {"params": filter(lambda p: p.requires_grad, vgg_face.parameters()), "lr": args.optim_lr}
    ]

    optimizer = optim.Adam(optimizer_dict, lr=args.optim_lr, betas=(0.9, 0.999), weight_decay=args.optim_weight_decay)

    
    criterion = {
        'ce_loss': nn.CrossEntropyLoss().to(device),
    }

    #Train Function
    start_epoch = 1
    for num_epoch in range(start_epoch, args.epochs+1):
        vgg_face.train()
        model.train()
        train_loss = 0
        i = 0
        pbar = tqdm(train_dataloader, leave=True)
        for batch in pbar:
            optimizer.zero_grad()

            #Feature extraction from VGG
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            features = vgg_face(images)

            #Sample pseudo negative sample
            if (i == 0):
                old_mean = torch.zeros(args.feature_dimension).to(device)
            else:
                old_mean = mean_vector
            mean_vector = torch.mean(features, axis=0)
            new_mean = args.alpha * old_mean + (1 - args.alpha) * mean_vector
            sampler = torch.distributions.multivariate_normal.MultivariateNormal(new_mean,args.std_dev * torch.eye(args.feature_dimension).to(device))
            noise = sampler.sample((features.shape[0],)).to(device)
            classifier_input = torch.cat([features, noise], dim=0)
            classifier_features, classifier_output = model(classifier_input)
            classifier_ground_truth = torch.cat([torch.zeros(features.shape[0]), torch.ones(noise.shape[0])], dim=0).to(device).long()

            #Loss
            tpc_loss = TPC_loss_hyp(classifier_features[:features.size(0)], c=args.curvature)
            classifier_loss = criterion['ce_loss'](classifier_output, classifier_ground_truth)
            loss = classifier_loss + tpc_loss
            loss.backward()
            train_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3)
            optimizer.step()

            i += 1
            pbar.set_description(f"Epochs {num_epoch}/{args.epochs}", refresh=True)
        train_loss /= i
        print(f"Epoch ({num_epoch}/{args.epochs}) | Total Train Loss: {train_loss}")
    
        ### Validation ###
        if (num_epoch % args.val_check_after_epoch == 0):
            with torch.no_grad():
                vgg_face.eval()
                model.eval()
                labels_list = []
                predictions_list = []
                val_pbar = tqdm(val_dataloader, leave=True)
                for batch in val_pbar:
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

                #Metrics and save models
                APCER, NPCER, ACER, EER, HTER, roc_auc, threshold, accuracy_threshold = statistics.calculate_metrics(labels_list, predictions_list)
                metrics["APCER"], metrics["NPCER"], metrics["ACER"], metrics["EER"], metrics["HTER"], metrics["ROC_AUC_Score"], metrics["Threshold"], metrics["Accuracy_threshold"] = APCER, NPCER, ACER, EER, HTER, roc_auc, threshold, accuracy_threshold
                print("APCER: ", metrics["APCER"]*100)
                print("NPCER: ", metrics["NPCER"]*100) 
                print("ACER: ", metrics["ACER"]*100)
                print("HTER: ", metrics["HTER"]*100)
                print("ROC_AUC_Score: ", metrics["ROC_AUC_Score"])
                print(f"Accuracy@{metrics['Threshold']}: ", metrics["Accuracy_threshold"]*100.0)
                if args.save_for_each_val_epoch == True:
                    filename = f"{args.save_root}/{args.dataset}/{args.expt_name}/epoch_{num_epoch}.pth"
                    save_checkpoint(num_epoch, vgg_face, model, metrics, filename)
                if metrics["HTER"] < best_HTER:
                    best_APCER = metrics["APCER"]
                    best_NPCER = metrics["NPCER"]
                    best_ACER = metrics["ACER"]
                    best_EER = metrics["EER"]
                    best_HTER = metrics["HTER"]
                    best_roc_auc = metrics["ROC_AUC_Score"]
                    best_threshold = metrics["Threshold"]
                    best_accuracy_threshold = metrics["Accuracy_threshold"]
                    filename = f"{args.save_root}/{args.dataset}/{args.expt_name}/best_epoch.pth"
                    save_checkpoint(num_epoch, vgg_face, model, metrics, filename)
                if num_epoch == args.epochs:
                    filename = f"{args.save_root}/{args.dataset}/{args.expt_name}/last_epoch.pth"
                    save_checkpoint(num_epoch, vgg_face, model, metrics, filename)

    sys.stdout = sys.__stdout__
    file.close()

if __name__ == '__main__':
    args = config.get_args()
    train(args)