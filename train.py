import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
from fcn8 import FCNs, VGGNet
from cardataset import Cardataset
import numpy as np
import time
import sys
import os

from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, './models')


def custom_loss(prediction, target, ratio):
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(ratio).cuda())
    loss_custom = crit(prediction, target)
    return loss_custom


def dice_loss(input, target):
    smooth = 1.

    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()

    return 1 - ((2. * intersection + smooth) /
                (iflat.sum() + tflat.sum() + smooth))


def weighted_cross_entropy_loss(prediction, target, weights=None):
    N, _, h, w = target.shape
    prediction = torch.clamp(prediction, min=1e-6, max=1 - 1e-6)
    prediction_0 = prediction[:, 0, :, :]
    prediction_1 = prediction[:, 1, :, :]
    target_0 = target[:, 0, :, :]
    target_1 = target[:, 1, :, :]
    # max_values, indices = torch.max(target, 1)
    # m_v, ind = torch.max(prediction, 1)
    # target = target.reshape([N, h, w])
    loss_0 = target_0 * torch.log(prediction_0) + (1 - target_0) * torch.log(1 - prediction_0)
    if weights is not None:
        assert len(weights) == 2

        # prediction[prediction == 0] = 1e-6
        # prediction[prediction == 1] = 1-1e-6

        loss_1 = weights[0] * (target_1 * torch.log(prediction_1)) + \
                 weights[1] * ((1 - target_1) * torch.log((1 - prediction_1)))
    # print(loss.data)
    else:
        loss = target * torch.log(prediction) + (1 - target) * torch.log(1 - prediction)
    return torch.neg(torch.mean(loss_0)) + torch.neg(torch.mean(loss_1))


def iou(pred, target):
    ious = []
    cls = 1
    # for cls in range(n_class):
    pred_inds = pred == cls
    target_inds = target == cls
    intersection = pred_inds[target_inds].sum()
    union = pred_inds.sum() + target_inds.sum() - intersection

    if union == 0:
        ious.append(float('nan'))  # if there is no ground truth, do not include in evaluation
    else:
        ious.append(float(intersection) / max(union, 1))
    return ious


def pixel_accuracy_foreground(pred, target):
    clas_1 = pred == 1
    tar_1 = target == 1
    correct = clas_1[tar_1].sum()
    total = tar_1.sum()
    return correct / total


def pixel_acc(pred, target):
    correct = (pred == target).sum()
    total = (target == target).sum()
    return correct / total


"""
 ------------------------------------------> network parameters------------------------------------------->
"""
n_class = 2
batch_size = 1
epochs = 150
lr = 1e-4
momentum = 0
w_decay = 1e-5
step_size = 15
gamma = 0.8
configs = "FCNs-BCEWithLogits_batch{}_epoch{}_sgd_scheduler-step{}-gamma{}_lr{}_momentum{}_w_decay{}".format(
    batch_size, epochs, step_size, gamma, lr, momentum, w_decay)
print("Configs:", configs)
# pos_weight=torch.tensor([7.62])
"""
 --------------------------------------> training data settings--------------------------------------->
"""
root_dir_train = "/home/uic55883/mapillary/training"
#root_dir_train = "/home/uic55883/semseg_vehicles"
image_folder = "images_subset2"
#image_folder = "trainingimages"
mask_folder = "grayinstances_subset2"
#mask_folder = "traininglabels"
"""
 ------------------------------------------->validation data------------------------------------------->
"""
root_dir_val = "/home/uic55883/mapillary/validation"
#root_dir_val = "/home/uic55883/semseg_vehicles/"
# root_dir_val = "/home/uic55883/subset/validation"
image_val = "validationimages"
mask_val = "validationlabels"

# create dir for model
model_dir = "models"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
model_path = os.path.join(model_dir, configs)

use_gpu = torch.cuda.is_available()
num_gpu = list(range(torch.cuda.device_count()))

"""
       --------------------------------> data loader part<------------------------------------------
"""

train_data = Cardataset(root_dir_train, image_folder, mask_folder, phase='train')
val_data = Cardataset(root_dir_val, image_val, mask_val, phase='val')
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
validation_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
PATH = "/home/uic41959/PycharmProjects/semanticsegmentation/"
vgg_model = VGGNet(requires_grad=True, remove_fc=True)

fcn_model = FCNs(pretrained_net=vgg_model, n_class=n_class)
if use_gpu:
    ts = time.time()
    vgg_model = vgg_model.cuda()
    fcn_model = fcn_model.cuda()
    print("Finish cuda loading, time elapsed {}".format(time.time() - ts))

weights = torch.FloatTensor([1, 15]).cuda()
pw = torch.reshape(weights, (1, 2, 1, 1)).cuda()
# class_weights = torch.FloatTensor(weights).cuda()

#criterion = nn.BCEWithLogitsLoss()  # 60           ## requires logits as input
criterion = nn.BCELoss()                          ###bceloss output from sigmoid
# ----------------sgd optimizer---------------
# optimizer = optim.SGD(fcn_model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
# ------------------end sgdoptimizer-----------
optimizer = optim.RMSprop(fcn_model.parameters(), lr=lr, momentum=momentum,
                          weight_decay=w_decay)  # scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size,
scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size,
                                gamma=gamma)  # decay LR by a factor of 0.5 every 30 epochs

# <<<<<<<<<------------------loading models ----------------------->>>>>>>>>>>>>>>>>>>>>>>>>

resume = '/home/uic55883/PycharmProjects/fcn/mapillarybceohnelogit19feb10.pt'
checkpoint = torch.load(resume)
# epoch = checkpoint['epoch']
fcn_model.load_state_dict(checkpoint['state_dict'])
optimizer.load_state_dict(checkpoint['optimizer'])
scheduler.load_state_dict(checkpoint['scheduler'])

# create dir for score
score_dir = os.path.join("scores", configs)
if not os.path.exists(score_dir):
    os.makedirs(score_dir)
IU_scores = np.zeros((200, n_class))
pixel_scores = np.zeros(200)

writer = SummaryWriter('runs/mapillarybceohnelogits19feb')


def train():
    for epoch in range(11, epochs):
        trainlosses = []
        running_loss = 0
        scheduler.step()

        ts = time.time()
        for iter, batch in enumerate(train_loader):
            optimizer.zero_grad()

            if use_gpu:

                inputs = Variable(batch['image'])
                inputs = inputs.float()
                inputs = inputs.cuda()
                labels = Variable(batch['mask'])
                labels = labels.float()
                # labels = labels.reshape([batch_size, 1, 640, 1024])
                labels = labels.cuda()
                target = batch['target']
                target = target.cuda()
                target = target.reshape([1, 1, 640, 1024])
            else:
                inputs, labels = Variable(batch['image']), Variable(batch['mask'])

            outputs = fcn_model(inputs)
            N, _, h, w = outputs.shape

            loss = criterion(outputs, labels)

            # loss = weighted_cross_entropy_loss(outputs, labels, weights=torch.tensor([20, 1]).cuda())

            trainlosses.append(loss.item())
            # loss=criterion(outputs.permute(0,2,3,1).view(-1,2),labels.permute(0,2,3,1).view(-1,2))
            running_loss += loss.item() * inputs.size(0)
            loss.backward()
            optimizer.step()
            if iter % 10 == 0:
                print("epoch{}, iter{}, loss: {}".format(epoch, iter, loss.item()))

        train_loss = running_loss / len(train_loader)
        writer.add_scalar('train/average_loss', train_loss, epoch)
        max_loss = np.max(trainlosses)
        writer.add_scalar('train/maximum_loss', max_loss, epoch)
        try:
            print("Finish epoch {}, time elapsed {}, epoch loss {}".format(epoch, time.time() - ts, train_loss))
        except Exception as e:
            print(e)
        torch.save({
            'epoch': epoch,
            'model': fcn_model,
            'state_dict': fcn_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }, 'mapillarybceohnelogit19feb%02d.pt' % epoch)
        # torch.save(fcn_model, 'fcn_fisheye_without%02d.pt'%epoch)
        val(epoch)


def val(epoch):
    fcn_model.eval()
    val_running_loss = 0
    validationlosses = []
    total_ious = []
    pixel_accs = []
    pixel_accs_1 = []
    for iter, batch in enumerate(validation_loader):
        with torch.no_grad():
            if use_gpu:
                inputs = Variable(batch['image'])
                inputs = inputs.float()
                inputs = inputs.cuda()
                labels = Variable(batch['mask'])
                labels = labels.float()
                labels = labels.cuda()
                # name=batch['name']
                # labels = labels.reshape([batch_size, 1, 640, 1024])
                target = batch['target'].cuda().reshape([batch_size, 1, 640, 1024])



            else:
                inputs = Variable(batch['mask'])

            out = fcn_model(inputs)
            N, _, h, w = out.shape
            # loss  = weighted_cross_entropy_loss(out, target, weights=[0.965,0.035])
            loss = criterion(out, labels)

            # loss = weighted_cross_entropy_loss(out, labels, weights=torch.tensor([20, 1]).cuda())
            validationlosses.append(loss.item())
            # loss = criterion(out.permute(0,2,3,1).view(-1,2), labels.permute(0,2,3,1).view(-1,2))
            val_running_loss += loss * inputs.size(0)

            #################--------------------------------->one class <-----------------------------------------------###########
            # pred=out
            # pred[pred > 0.8] = 1
            # pred[pred < 0.8] = 0
            #out = torch.sigmoid(out)
            ##############-----------------------------------------------------------------------------------------------###########
            pred = out.permute(0, 2, 3, 1).reshape(-1, 2).argmax(axis=1).reshape(N, h, w).cpu().numpy()
            # pred = pred.reshape([N,h,w]).cpu().numpy()
            target = target.reshape([N, h, w]).cpu().numpy().astype(int)
            for p, t in zip(pred, target):
                total_ious.append(iou(p, t))
                pixel_accs.append(pixel_acc(p, t))
            if iter % 10 == 0:
                print("validationepoch{}, iter{}, loss: {}".format(epoch, iter, loss.item()))

    val_loss = val_running_loss / len(validation_loader)
    max_loss_v = np.max(validationlosses)
    print("validationepoch{}, iter{}, loss: {}".format(epoch, iter, val_loss))
    writer.add_scalar('val_loss/average_loss', val_loss, epoch)
    writer.add_scalar('val_loss/max_loss', max_loss_v, epoch)
    # Calculate average IoU
    total_ious = np.array(total_ious).T  # n_class * val_len
    ious = np.nanmean(total_ious, axis=1)
    writer.add_scalar("ious", ious, epoch)
    pixel_accs = np.array(pixel_accs).mean()
    # t_foreground_accuracy=np.array(pixel_accs_1).T
    # foreground_accuracy=np.nanmean(t_foreground_accuracy,axis=1)
    print("epoch{}, pix_acc: {}, meanIoU: {}, IoUs: {}, ".format(epoch, pixel_accs, np.nanmean(ious), ious))
    writer.add_scalar("pixel_accuracy", pixel_accs, epoch)
    IU_scores[epoch] = ious
    np.save(os.path.join(score_dir, "meanIU"), IU_scores)
    pixel_scores[epoch] = pixel_accs
    np.save(os.path.join(score_dir, "meanPixel"), pixel_scores)


if __name__ == "__main__":
    #val(0)
     train()
