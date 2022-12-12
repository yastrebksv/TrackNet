from model import BallTrackerNet
import torch
from datasets import trackNetDataset
import torch.optim as optim
import os
from tensorboardX import SummaryWriter
from general import train, validate
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    parser.add_argument('--exp_id', type=str, default='default', help='path to saving results')
    parser.add_argument('--num_epochs', type=int, default=500, help='total training epochs')
    parser.add_argument('--lr', type=float, default=1.0, help='learning rate')
    parser.add_argument('--val_intervals', type=int, default=5, help='number of epochs to run validation')
    parser.add_argument('--steps_per_epoch', type=int, default=200, help='number of steps per one epoch')
    args = parser.parse_args()
    
    train_dataset = trackNetDataset('train')
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True
    )
    
    val_dataset = trackNetDataset('val')
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )    
    
    model = BallTrackerNet()
    device = 'cuda'
    model = model.to(device)
    
    exps_path = './exps/{}'.format(args.exp_id)
    tb_path = os.path.join(exps_path, 'plots')
    if not os.path.exists(tb_path):
        os.makedirs(tb_path)
    log_writer = SummaryWriter(tb_path)
    model_last_path = os.path.join(exps_path, 'model_last.pt')
    model_best_path = os.path.join(exps_path, 'model_best.pt')

    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    val_best_metric = 0

    for epoch in range(args.num_epochs):
        train_loss = train(model, train_loader, optimizer, device, epoch, args.steps_per_epoch)
        print('train loss = {}'.format(train_loss))
        log_writer.add_scalar('Train/training_loss', train_loss, epoch)
        log_writer.add_scalar('Train/lr', optimizer.param_groups[0]['lr'], epoch)

        if (epoch > 0) & (epoch % args.val_intervals == 0):
            val_loss, precision, recall, f1 = validate(model, val_loader, device, epoch)
            print('val loss = {}'.format(val_loss))
            log_writer.add_scalar('Val/loss', val_loss, epoch)
            log_writer.add_scalar('Val/precision', precision, epoch)
            log_writer.add_scalar('Val/recall', recall, epoch)
            log_writer.add_scalar('Val/f1', f1, epoch)
            if f1 > val_best_metric:
                val_best_metric = f1
                torch.save(model.state_dict(), model_best_path)           
            torch.save(model.state_dict(), model_last_path)
