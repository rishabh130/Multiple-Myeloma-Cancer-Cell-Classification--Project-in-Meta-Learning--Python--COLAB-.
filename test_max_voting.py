import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import f1_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
import os
import random
import glob
import sys
import torchvision.transforms.functional as TF
from scipy import ndimage, misc
import numpy as np
import cv2
import time
from torch.autograd import Variable
from vae import*

def warn(*args, **kwargs):
        pass
import warnings
warnings.warn = warn


size=350
class BasicDataset(Dataset):
    def __init__(self,folder_names, validation = False, patch_size=350):
        self.pos_images = list()
        self.neg_images = list()

        for folder in folder_names:
            print (folder)
            for fold in os.listdir(folder):
                print (fold)
                if "neoplastic" in fold:
                    self.pos_images += glob.glob(os.path.join(folder,fold)+"/*/*/*")
                else:
                    self.neg_images += glob.glob(os.path.join(folder,fold)+"/*/*/*")


        if not validation:
            try:
                over_sampled_neg_images = random.sample(self.neg_images, len(self.pos_images)- len(self.neg_images))
                self.neg_images += over_sampled_neg_images

            except:
                over_sampled_neg_images = random.sample(self.neg_images*2, len(self.pos_images)- len(self.neg_images))
                self.neg_images += over_sampled_neg_images
            self.custom_transform = transforms.Compose([ transforms.ToPILImage(),
                                                    transforms.CenterCrop(patch_size),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.RandomVerticalFlip(),
                                                    transforms.RandomRotation(360),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize([0.1001218,0.05760264, 0.19389437], [0.15300524, 0.08986596, 0.2886852])
                                                     ])
        else:
            self.custom_transform = transforms.Compose([ transforms.ToPILImage(),
                                                    transforms.CenterCrop(patch_size),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize([0.1001218,0.05760264, 0.19389437], [0.15300524, 0.08986596, 0.2886852])
                                                    ])
        self.num_pos_images = len(self.pos_images)
        self.num_neg_images = len(self.neg_images)


    def __len__(self):
        return self.num_pos_images + self.num_neg_images

    def __getitem__(self, idx):

        if idx >= self.num_pos_images:
            label = 0
            image_name = self.neg_images[idx-self.num_pos_images]
        else:
            label = 1
            image_name = self.pos_images[idx]

        image = imread(image_name)
        image_size=np.shape(image)
        # row_pad=int(image_size[0]-size)//2
        image=cv2.copyMakeBorder(image,(size-image_size[0])//2+(size-image_size[0])%2,(size-image_size[0])//2,(size-image_size[1])//2+(size-image_size[1])%2,(size-image_size[1])//2,cv2.BORDER_CONSTANT,value=[0,0,0])
        # print(image.shape)
        # image=image
        image = self.custom_transform(image)

        return image, label, image_name

def copy_data(m, i, o):
        my_embedding.copy_(o)

def train(val_folders, fold):
    file_name='max_voting_custom_prototype_bce_pro_dv7.csv'
    label_file_name='labels_custom_prototype_bce_pro_dv7.csv'


    file_name_bce='max_voting_custom_prototype_bce_bce_dv7.csv'
    label_file_name_bce='labels_custom_prototype_bce_bce_dv7.csv'


    with open(file_name, 'w') as f:
                        # f.write('fold0,fold1, fold2, fold3,fold4\n')
                         f.write('{0},{1},{2},{3},{4},{5}\n'.format('fold','weighted F1', 'class weighted F1', 'balanced accuracy','sensitivity','specificity'))

    with open(label_file_name, 'w') as f:
                        # f.write('fold0,fold1, fold2, fold3,fold4\n')
                        f.write('{0},{1},{2},{3},{4},{5},{6},{7}, {8}, {9}, {10}, {11}, {12}, {13}\n'.format('index','image_name', 'model 0','model 0 score', 'model 1', 'model 1 score','model 2', 'model 2 score','model 3', 'model 3 score','model 4', 'model 4 score','Majority', 'True'))

    with open(file_name_bce, 'w') as f:
                        # f.write('fold0,fold1, fold2, fold3,fold4\n')
                         f.write('{0},{1},{2},{3},{4},{5}\n'.format('fold','weighted F1', 'class weighted F1', 'balanced accuracy','sensitivity','specificity'))

    with open(label_file_name_bce, 'w') as f:
                        # f.write('fold0,fold1, fold2, fold3,fold4\n')
                        f.write('{0},{1},{2},{3},{4},{5},{6},{7}, {8}, {9}, {10}, {11}, {12}, {13}\n'.format('index','image_name', 'model 0','model 0 score', 'model 1', 'model 1 score','model 2', 'model 2 score','model 3', 'model 3 score','model 4', 'model 4 score','Majority', 'True'))



    model=classifier_mm_prelu_prototype_bce_v2()

    all_weighte_f1=[]
    balanced_acc=[]
    class_wf1=[]
    val_dataset = BasicDataset(val_folders, validation=True)
    test_loader = DataLoader(val_dataset,
                                       batch_size=1, shuffle=False, num_workers=8)
    dataset_test_len=len(val_dataset)
    index=-1;
    total_models=5
    initial_labels=[]
    true_labels=[]
    updated_labels=[]
    initial_labels=np.random.rand(dataset_test_len,total_models)
    label_prob=np.random.rand(dataset_test_len,total_models)

    initial_labels_bce=np.random.rand(dataset_test_len,total_models)
    label_prob_bce=np.random.rand(dataset_test_len,total_models)

    set_gpu=1
    majority_voting_labels=[]
    majority_voting_labels_bce=[]

    majority_all_weighte_f1=[]
    majority_all_weighte_f1_bce=[]

    majority_balanced_acc=[]
    majority_balanced_acc_bce=[]

    majority_class_wf1=[]
    majority_class_wf1_bce=[]
    pivot=0
    start_time = time.time()
    for (image, label,img_name)  in test_loader:

        index=index+1
        shift_index=-1
        true_labels.append(label.cpu().data.numpy()[0])
        for fold in range(total_models):
            shift_index=shift_index+1
        

            if fold==0:
                model.load_state_dict(torch.load('dict_models/model_'+ str(fold) + '_stage_2'))
            if fold==1:
              
                model.load_state_dict(torch.load('dict_models/model_'+ str(fold) + '_stage_2'))
            elif fold==2:
          
                model.load_state_dict(torch.load('dict_models/model_'+ str(fold) + '_stage_2'))
            elif fold==3:
              
                model.load_state_dict(torch.load('dict_models/model_'+ str(fold) + '_stage_2'))
            elif fold==4:
            
                model.load_state_dict( torch.load('dict_models/model_'+ str(fold) + '_stage_2'))

            model.cuda(1)
            model.eval()

            print('working on model {} and {} image'.format(fold,index))
            # print(img_name[0])

            with torch.no_grad():
                    image2, label = Variable(image.cuda(set_gpu)), Variable(label.cuda(set_gpu))
                    features, centers,distance,test_outputs, x_bce= model(image2)
                    # print(test_outputs)
                    _, predicted_test = torch.max(test_outputs.data, 1)
                    _, predicted_test_bce = torch.max(x_bce, 1)
                    
                    initial_labels[index, shift_index]=(predicted_test.cpu().data.numpy()[0])
                    label_prob[index, shift_index]=torch.exp(test_outputs.data).cpu().data.numpy()[0][int(initial_labels[index, shift_index])]

                    initial_labels_bce[index, shift_index]=(predicted_test_bce.cpu().data.numpy()[0])
                    label_prob_bce[index, shift_index]=torch.exp(x_bce.data).cpu().data.numpy()[0][int(initial_labels_bce[index, shift_index])]

            # print(initial_labels[index])
        if sum(initial_labels[index])>=3:
            majority_voting_labels.append(1)
        else:
            majority_voting_labels.append(0)

        if sum(initial_labels_bce[index])>=3:
            majority_voting_labels_bce.append(1)
        else:
            majority_voting_labels_bce.append(0)

        with open(label_file_name, 'a') as f:
                    im_name=img_name[0].split('/')[-1]
                    # f.write('fold0,fold1, fold2, fold3,fold4\n')
                    f.write('{0},{1},{2},{3},{4},{5},{6},{7}, {8}, {9}, {10}, {11}, {12}, {13}\n'.format(index,im_name, int(initial_labels[index, 0]), label_prob[index, 0], int(initial_labels[index, 1]), label_prob[index, 1],
                    int(initial_labels[index, 2]), label_prob[index, 2], int(initial_labels[index, 3]), label_prob[index, 3], int(initial_labels[index, 4]), label_prob[index, 4], int(majority_voting_labels[index]), true_labels[index]))



        with open(label_file_name_bce, 'a') as f:
                    im_name=img_name[0].split('/')[-1]
                    # f.write('fold0,fold1, fold2, fold3,fold4\n')
                    f.write('{0},{1},{2},{3},{4},{5},{6},{7}, {8}, {9}, {10}, {11}, {12}, {13}\n'.format(index,im_name, int(initial_labels_bce[index, 0]), label_prob_bce[index, 0], int(initial_labels_bce[index, 1]), label_prob_bce[index, 1],
                    int(initial_labels_bce[index, 2]), label_prob_bce[index, 2], int(initial_labels_bce[index, 3]), label_prob_bce[index, 3], int(initial_labels_bce[index, 4]), label_prob_bce[index, 4], int(majority_voting_labels_bce[index]), true_labels[index]))





    time_elapsed=time.time() - start_time

    majority_all_weighte_f1.append(f1_score(true_labels, majority_voting_labels, average='weighted'))
    majority_class_wf1.append(f1_score(true_labels, majority_voting_labels, average=None))
    majority_balanced_acc.append(balanced_accuracy_score(true_labels, majority_voting_labels))
    tn, fp, fn, tp = confusion_matrix(true_labels, majority_voting_labels).ravel()
    tn=(1.0*tn)
    fp=(1.0*fp)
    fn=(1.0*fn)
    tp=(1.0*tp)

    specificity=(tn)/(tn+fp)
    sensitivity=tp/(tp+fn)

    majority_all_weighte_f1_bce.append(f1_score(true_labels, majority_voting_labels_bce, average='weighted'))
    majority_class_wf1_bce.append(f1_score(true_labels, majority_voting_labels_bce, average=None))
    majority_balanced_acc_bce.append(balanced_accuracy_score(true_labels, majority_voting_labels_bce))

    tnbc, fpbc, fnbc, tpbc = confusion_matrix(true_labels, majority_voting_labels_bce).ravel()
    tnbc=(1.0*tnbc)
    fpbc=(1.0*fpbc)
    fnbc=(1.0*fnbc)
    tpbc=(1.0*tpbc)

    specificitybc=tnbc/(tnbc+fpbc)
    sensitivitybc=tpbc/(tpbc+fnbc)

    with open(file_name, 'a') as f:
                        # f.write('fold0,fold1, fold2, fold3,fold4,fold5,fold6\n')
                        f.write('{0},{1},{2},{3},{4},{5}\n'.format('majority voting',majority_all_weighte_f1[0], majority_class_wf1[0], majority_balanced_acc[0],sensitivity,specificity))
    
    with open(file_name_bce, 'a') as f:
                        # f.write('fold0,fold1, fold2, fold3,fold4,fold5,fold6\n')
                        f.write('{0},{1},{2},{3},{4},{5}\n'.format('majority voting',majority_all_weighte_f1_bce[0], majority_class_wf1_bce[0], majority_balanced_acc_bce[0],sensitivitybc,specificitybc))
    

    all_weighte_f1=[]
    balanced_acc=[]
    class_wf1=[]



    all_weighte_f1_bce=[]
    balanced_acc_bce=[]
    class_wf1_bce=[]

    for k2 in range(total_models):
        all_weighte_f1.append(f1_score(true_labels, initial_labels[:,k2], average='weighted'))
        class_wf1.append(f1_score(true_labels, initial_labels[:,k2], average=None))
        balanced_acc.append(balanced_accuracy_score(true_labels, initial_labels[:,k2]))

        tn, fp, fn, tp = confusion_matrix(true_labels,  initial_labels[:,k2]).ravel()
        tn=(1.0*tn)
        fp=(1.0*fp)
        fn=(1.0*fn)
        tp=(1.0*tp)
        specificity=tn/(tn+fp)
        sensitivity=tp/(tp+fn)

        all_weighte_f1_bce.append(f1_score(true_labels, initial_labels_bce[:,k2], average='weighted'))
        class_wf1_bce.append(f1_score(true_labels, initial_labels_bce[:,k2], average=None))
        balanced_acc_bce.append(balanced_accuracy_score(true_labels, initial_labels_bce[:,k2]))

        tnbc, fpbc, fnbc, tpbc = confusion_matrix(true_labels,  initial_labels_bce[:,k2]).ravel()

        tnbc=(1.0*tnbc)
        fpbc=(1.0*fpbc)
        fnbc=(1.0*fnbc)
        tpbc=(1.0*tpbc)
        specificitybc=tnbc/(tnbc+fpbc)
        sensitivitybc=tpbc/(tpbc+fnbc)

        with open(file_name, 'a') as f:
                            # f.write('fold0,fold1, fold2, fold3,fold4,fold5,fold6\n')
                            f.write('{0},{1},{2},{3},{4},{5}\n'.format(k2,all_weighte_f1[k2], class_wf1[k2], balanced_acc[k2],sensitivity,specificity))


        with open(file_name_bce, 'a') as f:
                            # f.write('fold0,fold1, fold2, fold3,fold4,fold5,fold6\n')
                            f.write('{0},{1},{2},{3},{4},{5}\n'.format(k2,all_weighte_f1_bce[k2], class_wf1_bce[k2], balanced_acc_bce[k2],sensitivitybc,specificitybc))

    with open(file_name, 'a') as f:
                            # f.write('fold0,fold1, fold2, fold3,fold4,fold5,fold6\n')
                            f.write('{0},{1},{2},{3},{4},{5}\n'.format('total_time',time_elapsed, time_elapsed, time_elapsed,time_elapsed,time_elapsed))




    print("Weighted F1 with initial labels is {}".format(all_weighte_f1))
    print("class F1 with initial labels is {}".format(class_wf1))
    print("bac with initial labels is {}".format(balanced_acc))
    #np.savetxt('fold'+str(fold)+'_rot.csv',initial_labels )




if __name__ == '__main__':
    best_val_acc = 0.
    fold=(sys.argv[1])
    ref_image_name = "./ref_all.bmp"

    folders = sorted(glob.glob("../../MM_dataset/new_MM_segmented_data/testing/*"))
    print(folders)
    val_folders = [folders[0]]

    print (val_folders)
    train(val_folders, fold)
