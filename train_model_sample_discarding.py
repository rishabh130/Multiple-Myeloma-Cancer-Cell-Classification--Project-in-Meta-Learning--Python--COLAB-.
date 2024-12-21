
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage.io import imread, imsave
import os
import random
import glob
# from tensorboard_logger import configure, log_value
# from Models import *
import sys
# import pandas as pd
from data_utils import *
from sklearn.metrics import f1_score
import cv2
import torchvision.models as models
from vae import*
import numpy as np
from torch.autograd import Variable
# torch.manual_seed(1234)



size=350

class BasicDataset(Dataset):
    def __init__(self,folder_names, validation = False, patch_size=350):
        self.pos_images = list()
        self.neg_images = list()

        for folder in folder_names:
            for fold in os.listdir(folder):

                if "neoplastic" in fold:

                    self.pos_images += glob.glob(os.path.join(folder,fold)+"/*/*/*")
                else:
                    print(fold)
                    self.neg_images += glob.glob(os.path.join(folder,fold)+"/*/*/*")
        print(len(self.pos_images),len(self.neg_images))

        if not validation:
            if len(self.pos_images) > len(self.neg_images):
                try:
                    over_sampled_neg_images = random.sample(self.neg_images, len(self.pos_images)- len(self.neg_images))
                    self.neg_images += over_sampled_neg_images

                except:
                    over_sampled_neg_images = random.sample(self.neg_images*2, len(self.pos_images)- len(self.neg_images))
                    self.neg_images += over_sampled_neg_images
            else:
                try:
                    over_sampled_pos_images = random.sample(self.pos_images, len(self.neg_images)- len(self.pos_images))
                    self.pos_images += over_sampled_pos_images

                except:
                    over_sampled_pos_images = random.sample(self.pos_images*2, len(self.neg_images)- len(self.pos_images))
                    self.pos_images += over_sampled_pos_images

            self.custom_transform = transforms.Compose([ transforms.ToPILImage(),
                                                    transforms.CenterCrop(patch_size),
                                                    #transforms.Resize(224),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.RandomVerticalFlip(),
                                                    transforms.RandomRotation(360),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize([0.1001218,0.05760264, 0.19389437], [0.15300524, 0.08986596, 0.2886852])
                                                     ])
        else:
            self.custom_transform = transforms.Compose([ transforms.ToPILImage(),
                                                    transforms.CenterCrop(patch_size),
                                                    #transforms.Resize(224),
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
        #row_pad=int(image_size[0]-size)//2
        image=cv2.copyMakeBorder(image,(size-image_size[0])//2+(size-image_size[0])%2,(size-image_size[0])//2,(size-image_size[1])//2+(size-image_size[1])%2,(size-image_size[1])//2,cv2.BORDER_CONSTANT,value=[0,0,0])
        # print(image.shape)
        image = self.custom_transform(image)

        return image, label, image_name

def write_to_tensorboard(epoch, loss, labels, outputs, val=False):
    labels = np.asarray(labels)
    outputs = np.array(outputs)

    assert len(labels) == len(outputs), "labels and outputs size doesn't match"

    TP = len(labels[np.logical_and(labels==1,outputs==1)])
    FP = len(labels[np.logical_and(labels==0,outputs==1)])
    TN = len(labels[np.logical_and(labels==0,outputs==0)])
    FN = len(labels[np.logical_and(labels==1,outputs==0)])

    eps = 1e-8
    precision = TP/float(TP + FP + eps)
    recall = TP/float(TP + FN + eps)
    precision_neg = TN/float(TN + FN+ eps)
    recall_neg = TN/float(TN + FP+ eps)
    f1 = 2*precision*recall/float(precision+recall + eps)
    f1_neg = 2*precision_neg*recall_neg/float(precision_neg+recall_neg + eps)
    accuracy = (TP+TN)/float(TP+TN+FP+FN + eps)

    if val:
        log_value("Loss/val_loss", loss, epoch)
        log_value("Precision/val_positive/", precision, epoch)
        log_value("Recall/val_positive", recall, epoch)
        log_value("F1/val_positive", f1, epoch)
        log_value("Precision/val_negetive", precision_neg, epoch)
        log_value("Recall/val_negetive", recall_neg, epoch)
        log_value("F1/val_negetive", f1_neg, epoch)
        log_value("Accuracy/val_accuracy", accuracy, epoch)

    else:
        log_value("Loss/train_loss", loss, epoch)
        log_value("Precision/train_positive/", precision, epoch)
        log_value("Recall/train_positive", recall, epoch)
        log_value("F1/train_positive", f1, epoch)
        log_value("Precision/train_negetive", precision_neg, epoch)
        log_value("Recall/train_negetive", recall_neg, epoch)
        log_value("F1/train_negetive", f1_neg, epoch)
        log_value("Accuracy/train_accuracy", accuracy, epoch)

    return accuracy


def lr_scheduler(optimizer, init_lr, epoch):


    for param_group in optimizer.param_groups:

        if  epoch >=0 and epoch <=10:
            param_group['lr']=1e-5

        if epoch>10 and epoch <=30:
            param_group['lr']=1e-6

        if epoch >30:
            param_group['lr']=1e-7

  

        print('Current learning rate is {}'.format(param_group['lr']))


    return optimizer

def train(folder_names, val_folders, num_epochs , val_folder, gpu_no):
    best_val_acc=0.0
    
  
    model = torch.load('initial_model.pt')

    
    model= model.cuda(gpu_no)
    print(model)

    lrate=1e-3
 

    optimizer_s=optim.Adam(model.parameters(), lr=lrate)


    train_dataset = BasicDataset(folder_names)
    val_dataset = BasicDataset(val_folders, validation=True)
    dataset_loader = DataLoader(train_dataset,
                                             batch_size=64, shuffle=True, num_workers=8)
    test_loader = DataLoader(val_dataset,
                                        batch_size=64, shuffle=False, num_workers=8)
    dataset_test_len=len(val_dataset)

    dataset_train_len=len(train_dataset)
    print(dataset_test_len)
    print(dataset_train_len)
 
    epochs= []
    train_acc=[]
    test_acc=[]
    train_loss=[]
    test_loss = []
    train_wf1=[]
    test_wf1=[]
    
    w = torch.Tensor([1, .5])
    w=w.cuda()
    label_flip_imgs=[]

    for epoch in range(num_epochs):
        model.train()
        counter=0
        epoch_loss = 0.
        all_labels = list()
        all_outputs = list()
        epochs.append(epoch)
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        optimizer  = lr_scheduler(optimizer_s, lrate, epoch)
        print('*' * 70)
        running_loss = 0.0
        running_corrects = 0.0
        running_corrects_bce = 0.0
        train_batch_ctr = 0.0
        GT_train=[]
        predicted_labels_train=[]
        
        for batch_num, (images, labels, image_name) in enumerate(dataset_loader):
     
            image_names=[]
            prob=[]
            prob_bce=[]
            images = Variable(images.cuda(gpu_no),requires_grad=True)
            labels = Variable(labels.cuda(gpu_no),requires_grad=False)

            all_ind=[]
            effective_ind=[]
            doubt_ind=[]

       

            for im_name in image_name:
                    image_names.append(im_name)

                
               
            features, centers,distance,outputs, x_bce = model(images)
    
            _, preds = torch.max(outputs, 1)

            _, preds_bce = torch.max(x_bce, 1)

            k=-1

            for o in outputs:

                k+=1
                all_ind.append(k)
                prob.append(torch.exp(o.data).cpu().data.numpy()[preds.cpu().data.numpy()[k]])
            k=-1
            for o_bce in x_bce:
                k+=1
                prob_bce.append(torch.exp(o_bce.data).cpu().data.numpy()[preds_bce.cpu().data.numpy()[k]])


            k=-1
            
            for p in prob:
                k+=1
                p1=(prob_bce[k])
                if preds[k] != preds_bce[k] and p<.99 and p1 <.99:
                    print('flipping label, original_label {}, prot_label: {} with p: {}, bce_label {} with p_bce: {}'.format(labels[k], preds[k], p, preds_bce[k], p1))
                    label_flip_imgs.append(image_names[k])
                   
                    doubt_ind.append(k)
                    counter+=1
            
            effective_ind=np.delete(all_ind, doubt_ind)

            images=images[effective_ind]
            labels=labels[effective_ind]

          

            features, centers,distance,outputs, x_bce = model(images)

            _, preds = torch.max(distance, 1)

            _, preds_bce = torch.max(x_bce, 1)


            optimizer.zero_grad()


            loss1 = F.nll_loss(outputs, labels)

               

            loss2=torch.pow(torch.dot(centers[:,0], centers[:,1]),2)

            loss3=F.nll_loss(x_bce, labels)


            loss=loss1+loss2+loss3

     
            loss.backward()
            optimizer.step()

            train_batch_ctr = train_batch_ctr + 1



            running_loss += loss.item()

            running_corrects += torch.sum(preds == labels.data)

            running_corrects_bce += torch.sum(preds_bce == labels.data)

            epoch_acc = float(running_corrects) / (dataset_train_len)

            epoch_acc_bce = float(running_corrects_bce) / (dataset_train_len)

            GT_train.append(labels.cpu().data.numpy()[0])
            predicted_labels_train.append(preds.cpu().data.numpy()[0])

            

        print('removed {} samples'.format(counter))
        print ('Train corrects: {} Train samples: {} Train accuracy: {} Train accuracy BCE: {}' .format( running_corrects, (dataset_train_len),epoch_acc, epoch_acc_bce))
        train_acc.append(epoch_acc)
        train_loss.append(running_loss / train_batch_ctr)
        # import pdb;pdb.set_trace()
        train_wf1.append(f1_score(GT_train,predicted_labels_train, average='weighted'))
        # train_error.append(float((dataset_train_len)-running_corrects) / float((dataset_train_len)))




        model.eval()
        test_running_corrects = 0.0
        test_running_corrects_bce = 0.0
        test_batch_ctr = 0.0
        test_running_loss = 0.0
        test_total = 0.0
        GT_test=[]
        predicted_labels_test=[]



        for (image, label,_)  in test_loader:



            with torch.no_grad():

                
                image, label = Variable(image.cuda()), Variable(label.cuda())



                features, centers,distance,test_outputs, x_bce = model(image)
                _, predicted_test = torch.max(distance, 1)
                _, predicted_test_bce = torch.max(x_bce, 1)


                loss1 = F.nll_loss(test_outputs, label)
            

                loss2=torch.pow(torch.dot(centers[:,0], centers[:,1]),2)

                loss3 = F.nll_loss(x_bce, label)


                loss=loss1+loss2+loss3

        

                test_running_loss += loss.item()
                test_batch_ctr = test_batch_ctr+1

                test_running_corrects += torch.sum(predicted_test == label.data)
                test_running_corrects_bce += torch.sum(predicted_test_bce == label.data)
                test_epoch_acc = float(test_running_corrects) / (dataset_test_len)
                test_epoch_acc_bce = float(test_running_corrects_bce) / (dataset_test_len)
                GT_test.append(label.cpu().data.numpy()[0])
                predicted_labels_test.append(predicted_test.cpu().data.numpy()[0])

        if test_epoch_acc > best_val_acc:
            torch.save(model, './5_fold_'+str(val_folder)+'_SD_best.pt')
            best_val_acc=test_epoch_acc

        test_acc.append(test_epoch_acc)
        test_loss.append(test_running_loss / test_batch_ctr)
        # test_wf1.append(float((dataset_test_len)-test_running_corrects) / float((dataset_test_len)))
        test_wf1.append(f1_score(GT_test,predicted_labels_test,average='weighted'))

        print('Test corrects: {} Test samples: {} Test accuracy {} Test accuracy BCE {}' .format(test_running_corrects,(dataset_test_len),test_epoch_acc, test_epoch_acc_bce))

        print('Train loss: {} Test loss: {}' .format(train_loss[epoch],test_loss[epoch]))

        print('Train error: {} Test error {}' .format(train_wf1[epoch],test_wf1[epoch]))

        print('*' * 70)



    torch.save(model, './5_fold_'+ str(val_folder)+'_SD_final.pt')



if __name__ == '__main__':
    best_val_acc = 0.0
 
    val_folder=sys.argv[1]
    gpu_no=int(sys.argv[2])


  
    folders = sorted(glob.glob("../new_MM_segmented_data/training/*"))
    print(folders)
    train_folders=[folders[0], folders[1], folders[2],folders[3], folders[4]]
    del train_folders[int(val_folder)]

    val_folders = [folders[int(val_folder)]]

    for k in range(len(train_folders)):
        print('training on fold:\n {}'.format(train_folders[k][-1]))
    print('validation on fold:\n {}'.format(val_folders[0][-1]))
    train(train_folders, val_folders,35, val_folder, gpu_no)


