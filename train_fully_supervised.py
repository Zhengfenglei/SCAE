import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from Amodel import SCAE_Net
import argparse
from torch.optim.lr_scheduler import MultiStepLR
from load_data import load_data
import os
from utils import *
from train_test_epoch import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default='0.01', type=float, help='lr')
    parser.add_argument('--lr_steps', default=[25, 50, 75], type=float,  help='decay learning rate')
    parser.add_argument('--batch_size', default='32', type=int, help='batch_size of training data')

    parser.add_argument('--label_ratio', default=1.0, type=float, help='labeled data ratio.')
    parser.add_argument('--result_path', default='./', type=str, help='Result directory path')
    parser.add_argument('--store_name', default='model', type=str, help='model save name')

    parser.add_argument('--begin_epoch', default=1, type=int, help='Training begins at this epoch.')
    parser.add_argument('--n_epochs', default=100, type=int, help='Number of total epochs to run')

    parser.add_argument('--manual_seed', default=2, type=int, help='Manually set random seed')

    args = parser.parse_args()
    torch.manual_seed(args.manual_seed)
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)

    model = SCAE_Net()
    model_dict = model.state_dict()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=args.lr_steps, gamma=0.1)
    criterion = nn.CrossEntropyLoss().cuda()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() >= 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model.to(device)
    train_loader, test_loader = load_data(args.batch_size)
    train_logger = Logger(
        os.path.join(args.result_path, 'train.log'),
        ['epoch', 'loss', 'prec1', 'lr'])
    train_batch_logger = Logger(
        os.path.join(args.result_path, 'train_batch.log'),
        ['epoch', 'batch', 'iter', 'loss', 'prec1', 'lr'])
    val_logger = Logger(
        os.path.join(args.result_path, 'val.log'), ['epoch', 'loss', 'prec1'])
    best_prec1 = 0.0

    for i in range(args.begin_epoch, args.n_epochs + 1):
        train_epoch(i, train_loader, model, criterion, optimizer, args,
                    train_logger, train_batch_logger)
        scheduler.step()
        torch.save(model.state_dict(), '%s/%s_checkpoint.pkl' % (args.result_path, args.store_name))
        validation_loss, prec1 = val_epoch(i, test_loader, model, criterion, args,
                                           val_logger)
        is_best = prec1 > best_prec1
        if is_best:
            best_prec1 = max(prec1, best_prec1)
            torch.save(model.state_dict(), '%s/%s_best.pkl' % (args.result_path, args.store_name))

    print('best_acc:', best_prec1)

