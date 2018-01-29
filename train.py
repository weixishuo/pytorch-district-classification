import torch
import torch.optim as optim
import torch.nn as nn
from model import Encoder
from dataLoader import ImageSet,readAll
from torch.autograd import Variable
import sys
import numpy as np
import os

def train(train_loader, val_loader, epochnum, save_path='.', save_freq=None):
    iter_size = len(train_loader)
    net = Encoder()
    net.cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=2e-4)

    for epoch in range(epochnum):
        print('epoch : {}'.format(epoch))
        net.train()
        train_loss = 0
        train_correct = 0
        total = 0
        net.training = True
        for i, data in enumerate(train_loader):
            sys.stdout.write('iter : {} / {}\r'.format(i, iter_size))
            sys.stdout.flush()
            #print('iter: {} / {}'.format(i, iter_size))
            inputs, labels = data
            inputs, labels = Variable(inputs.cuda()), labels.cuda()
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs,  Variable(labels))
            loss.backward()
            optimizer.step()
            train_loss += loss.data[0]
            pred = (torch.max(outputs.data, 1)[1])
            train_correct += (pred==labels).sum()
            total += labels.size(0)
        sys.stdout.write(' ' * 20 + '\r')
        sys.stdout.flush()

        print('train_loss:{}, train_acc:{:.2%}'.format(train_loss / total,
                                                       train_correct / total))
        val_loss = 0
        val_correct = 0
        total = 0
        net.training = False
        for data in val_loader:
            net.eval()
            inputs, labels = data
            inputs, labels = Variable(inputs).cuda(), labels.cuda()
            outputs = net(inputs)
            pred = torch.max(outputs.data, 1)[1]
            total += labels.size(0)
            loss = criterion(outputs, Variable(labels))
            val_loss += loss.data[0]
            val_correct += (pred==labels).sum()

        print('val_loss:{}, val_acc:{:.2%}'.format(val_loss / total, val_correct / total))
        optimizer.param_groups[0]['lr'] *= np.exp(-0.4)
        if save_freq and epoch % save_freq == save_freq - 1:
            net_name = os.path.join(save_path, 'epoch_{}'.format(epoch))
            torch.save(net, net_name)
    torch.save(net, os.path.join(save_path, 'trained_net'))

if __name__ == "__main__":
    val_num = 3000
    entries = readAll('/data/public/weixishuo/face_classification/2018_01_05/face_labels.txt')
    train_set = ImageSet(entries[0:-val_num])
    val_set = ImageSet(entries[-val_num:])
    train_loader = torch.utils.data.DataLoader(train_set,batch_size=64,
                                               shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=64,
                                             shuffle=True, num_workers=2)

    train(train_loader, val_loader, 30,
          save_freq=5, save_path='/data/public/weixishuo/image-caption/models_20180129')
