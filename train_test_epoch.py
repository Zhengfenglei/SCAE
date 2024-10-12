from utils import *
import time
from torch.autograd import Variable
import os

def train_epoch(epoch, data_loader, model, criterion, optimizer, opt,
                epoch_logger, batch_logger):
    model.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()

    for i, (inputs, targets) in enumerate(data_loader):
        targets = targets.cuda()
        inputs = inputs.cuda()

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        losses.update(loss.data, inputs.size(0))
        prec1, prec3 = calculate_accuracy(outputs.data, targets.data, topk=(1, 3))
        top1.update(prec1, inputs.size(0))
        top3.update(prec3, inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epoch_logger.log({
        'epoch': epoch,
        'loss': losses.avg.item(),
        'prec1': top1.avg.item(),
        'lr': optimizer.param_groups[0]['lr']
    })
    print('\nTrain Epoch: [{}] Loss {:.4f} Prec@1 {:.5f} '.format(
        epoch,
        losses.avg.item(),
        top1.avg.item(),
        ))


def val_epoch(epoch, data_loader, model, criterion, opt, logger):
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    for i, (inputs, targets) in enumerate(data_loader):
        targets = targets.cuda()
        with torch.no_grad():
            inputs = inputs.cuda()
            inputs = Variable(inputs)
            targets = Variable(targets)
        outputs = model(inputs)

        loss = criterion(outputs, targets)
        prec1, prec5 = calculate_accuracy(outputs.data, targets.data, topk=(1, 3))
        top1.update(prec1, inputs.size(0))
        top5.update(prec5, inputs.size(0))
        losses.update(loss.data, inputs.size(0))

    logger.log({'epoch': epoch,
                'loss': losses.avg.item(),
                'prec1': top1.avg.item(),
                })
    print('Test Epoch: [{}] Loss {:.4f} Prec@1 {:.5f} '.format(
        epoch,
        losses.avg.item(),
        top1.avg.item(),
    ))

    return losses.avg.item(), top1.avg.item()

def train_AE_epoch(epoch, data_loader, model, criterion, optimizer, opt,
                epoch_logger, batch_logger):
    model.train()
    losses = AverageMeter()
    batch_time = AverageMeter()
    end_time = time.time()

    for i, (inputs, targets) in enumerate(data_loader):
        inputs = inputs.cuda()
        optimizer.zero_grad()
        _, outputs = model(inputs)
        loss = criterion(outputs, inputs)
        losses.update(loss.data, inputs.size(0))
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end_time)
        end_time = time.time()
        batch_logger.log({
            'epoch': epoch,
            'batch': i + 1,
            'iter': (epoch - 1) * len(data_loader) + (i + 1),
            'batch_time': batch_time.val
        })

    epoch_logger.log({
        'epoch': epoch,
        'loss': losses.avg.item(),
        'lr': optimizer.param_groups[0]['lr'],
    })
    print('\nTrain Epoch: [{}] Loss {:.8f}'.format(
        epoch,
        losses.avg.item(),
        ))

