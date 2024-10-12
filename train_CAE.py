import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from model import AutoEncoder
import argparse
from torch.optim.lr_scheduler import MultiStepLR
from load_data import *
import os
from utils import *
from train_test_epoch import *
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default='0.01', type=float, help='lr')
    parser.add_argument('--lr_steps', default=[25, 50, 75, 150], type=float,  help='decay learning rate')
    parser.add_argument('--batch_size', default='32', type=int, help='batch_size of training data')

    parser.add_argument('--result_path', default='./results/', type=str, help='Result directory path')
    parser.add_argument('--store_name', default='model', type=str, help='Result directory path')

    parser.add_argument('--begin_epoch', default=1, type=int, help='Training begins at this epoch.')
    parser.add_argument('--n_epochs', default=200, type=int, help='Number of total epochs to run')

    parser.add_argument('--manual_seed', default=2, type=int, help='Manually set random seed')

    args = parser.parse_args()
    torch.manual_seed(args.manual_seed)
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)

    model = AutoEncoder()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=args.lr_steps, gamma=0.1)
    criterion = nn.MSELoss().cuda()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() >= 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model.to(device)

    train_loader = load_data(args.batch_size)

    train_logger = Logger(
        os.path.join(args.result_path, 'train.log'),
        ['epoch', 'loss', 'lr'])
    train_batch_logger = Logger(
        os.path.join(args.result_path, 'train_batch.log'),
        ['epoch', 'batch', 'iter', 'batch_time'])

    torch.cuda.synchronize()
    start_time = time.time()
    for i in range(args.begin_epoch, args.n_epochs + 1):
        train_AE_epoch(i, train_loader, model, criterion, optimizer, args,
                    train_logger, train_batch_logger)
        scheduler.step()
        torch.save(model.state_dict(), '%s/%s_checkpoint.pkl' % (args.result_path, args.store_name))
    torch.cuda.synchronize()
    train_time = time.time() - start_time
    print("time cost:", train_time)
