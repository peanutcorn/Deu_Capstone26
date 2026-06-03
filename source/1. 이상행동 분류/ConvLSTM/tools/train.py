import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau

import time
import os

import _init_paths
from models.NIA_LSTM import LSTM_NIA
from dataset.NIADataset import NIADataset

# hyperparameters
begin_epoch = 0
end_epoch = 50
batch_size = 4
learning_rate = 1e-3

output_dir = 'output'

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# model
model = LSTM_NIA(
    hidden_size = 256,
    num_classes = 8,
    num_layers = 1,
    backbone = 'resnet50',
    device = device
)
#model = model.to(device)
model = torch.nn.DataParallel(model, device_ids=(0,1,2,3)).to(device)


# data loading
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)

tmp_path = '/workspace/home/mspark/DB/NIA'
print("====> exists : {}".format(os.path.exists(tmp_path)))
# train_dataset = NIADataset(
#     root = '/mnt/f/NIA',
#     image_set = 'train',
#     fps = 3,
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         normalize
#     ])
# )
train_dataset = NIADataset(
    root = '/workspace/home/mspark/DB/NIA',
    image_set = 'train',
    fps = 3,
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
)

# val_dataset = NIADataset(
#     root = '/mnt/f/NIA',
#     image_set = 'val',
#     fps = 3,
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         normalize
#     ])
# )
val_dataset = NIADataset(
    root = '/workspace/home/mspark/DB/NIA',
    image_set = 'val',
    fps = 3,
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
)



# data loader
train_loader = torch.utils.data.DataLoader(
    dataset = train_dataset,
    batch_size = batch_size,
    shuffle = True,
    num_workers = 4
)

val_loader = torch.utils.data.DataLoader(
    dataset = val_dataset,
    batch_size = batch_size,
    shuffle = False,
    num_workers = 4
)


# optimizer
optimizer = torch.optim.Adam(
    params=model.parameters(), 
    lr=learning_rate
)


# loss
data_distribution = train_dataset.get_distribution()
weights = [1 - (x / sum(data_distribution)) for x in data_distribution]
weights = torch.FloatTensor(weights).to(device)

criterion = nn.CrossEntropyLoss(weights).to(device)


# scheduler 
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, end_epoch, eta_min=learning_rate*0.1, last_epoch=begin_epoch-1)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
        
        

def train(model, train_loader, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()
    
    end = time.time()
    for i, (input, label) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        
        input = input.to(device)
        # compute output
        output = model(input)        
        label = torch.LongTensor(label).to(device)

        # loss
        loss = criterion(output, label)
        
        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # measure accuracy ans record loss
        losses.update(loss.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % 100 == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses
                  )
            print(msg)
                      
                      
def validation(model, val_loader, val_dataset, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    
    model.eval()
    
    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, 1), dtype=np.float32
    )

    with torch.no_grad():
        end = time.time()
        
        for i, (input, label) in enumerate(val_loader):
            # compute output
            input = input.to(device)
            output = model(input)
            label = torch.LongTensor(label).to(device)
            
            # loss 
            loss = criterion(output, label)
            
            num_images = input.size(0)
            losses.update(loss.item(), num_images)
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            if i % 100 == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses)
                print(msg)


if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# train
for epoch in range(begin_epoch, end_epoch):
    train(
        model = model,
        train_loader = train_loader,
        criterion = criterion,
        optimizer = optimizer,
        epoch = epoch
    )
    
    validation(
        model = model,
        val_dataset = val_dataset,
        val_loader = val_loader,
        criterion = criterion
    )
    
    # save
    states = {
        'epoch' : epoch + 1,
        'name' : 'LSTM',
        'state_dict' : model.state_dict(),
        'optimizer' : optimizer.state_dict()
    }
    save_name = os.path.join(output_dir, 'model_' + str(epoch+1) + '.pth')
    torch.save(states, save_name)
    print("===> save file : {}".format(save_name))
