"""
@Brief: Train the key frame transmission network
"""

import torch
import torch.nn as nn
import argparse
import torchvision
from dataset_image import UCF101_Image
from torch.utils.data import DataLoader
from WITT import Witt


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main(args):

    train_data = UCF101_Image('../data/UCF101/UCF-101-TRA-IMAGE')
    valid_data = UCF101_Image('../data/UCF101/UCF-101-TRA-IMAGE')
    train_data_loader = DataLoader(train_data, batch_size=args.batch, shuffle=True, num_workers=0, drop_last=True)
    valid_data_loader = DataLoader(valid_data, batch_size=args.batch, shuffle=True, num_workers=0, drop_last=True)

    # model
    model = Witt(args.batch, (240, 320), C=8).float().to(device)

    loss_fn = torch.nn.MSELoss().to(device)  # MSE loss function
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(adam_optimizer, milestones=[50, 100, 140, 180], gamma=0.3)

    epochs = args.epochs
    train_step = 0

    loss = 0
    record_loss = 0
    record_valid_losss = 0

    loss_per_epoch = 0
    loss_valid_per_epoch = 0
    epoch_count = 0
    epoch_valid_count = 0

    for epoch in range(epochs):

        print("========================================================")
        print("Start of the {}th training round, learning rate {}".format(epoch + 1, adam_optimizer.state_dict()['param_groups'][0]['lr']))
        for src_img in train_data_loader:
            src_img = src_img.to(device)

            model.train()
            compressed_img = model(src_img)

            loss = loss_fn(src_img, compressed_img)
            loss_per_epoch += loss.item()
            adam_optimizer.zero_grad()
            epoch_count += 1
            loss.backward()
            adam_optimizer.step()

            train_step += 1
            if train_step % 25 == 0:
                print("Amount of data trained in the current round: {}, Loss for the current batch: {}".format(train_step*args.batch, loss.item()))

                model.eval()
                total_valid_loss = 0
                with torch.no_grad():
                    valid_count = 0
                    for src_img_valid in valid_data_loader:
                        if valid_count < 5:
                            src_img_valid = src_img_valid.to(device)

                            compressed_img_valid = model(src_img_valid)

                            valid_loss = loss_fn(src_img_valid, compressed_img_valid)
                            total_valid_loss += valid_loss.item()
                            valid_count += 1
                        else:
                            break
                    print("Average loss on the validation set: {}".format(total_valid_loss / 5))
                    print("----------------------------")
                loss_valid_per_epoch += total_valid_loss / 5
                epoch_valid_count += 1
        scheduler.step()
        train_step = 0

        valid_losss = loss_valid_per_epoch/epoch_valid_count

        if epoch == 0:
            record_valid_losss = valid_losss
            torch.save(model.encoder.state_dict(),
                       "../checkpoints/temp/enc_epoch{}_loss{:.2f}.pth".format(epoch + 1, valid_losss))
            torch.save(model.q.state_dict(),
                       "../checkpoints/temp/q_epoch{}_loss{:.2f}.pth".format(epoch + 1, valid_losss))
            torch.save(model.dq.state_dict(),
                       "../checkpoints/temp/dq_epoch{}_loss{:.2f}.pth".format(epoch + 1, valid_losss))
            torch.save(model.decoder.state_dict(),
                       "../checkpoints/temp/dec_epoch{}_loss{:.2f}.pth".format(epoch + 1, valid_losss))
        elif valid_losss < record_valid_losss:
            torch.save(model.encoder.state_dict(),
                       "../checkpoints/temp/enc_epoch{}_loss{:.2f}.pth".format(epoch + 1, valid_losss))
            torch.save(model.q.state_dict(),
                       "../checkpoints/temp/q_epoch{}_loss{:.2f}.pth".format(epoch + 1, valid_losss))
            torch.save(model.dq.state_dict(),
                       "../checkpoints/temp/dq_epoch{}_loss{:.2f}.pth".format(epoch + 1, valid_losss))
            torch.save(model.decoder.state_dict(),
                       "../checkpoints/temp/dec_epoch{}_loss{:.2f}.pth".format(epoch + 1, valid_losss))
            record_valid_losss = valid_losss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', default=8, type=int)
    parser.add_argument('--epochs', default=200, type=int)

    args_ = parser.parse_args()

    main(args_)


