
import torch
import torch.nn as nn
import argparse
import torchvision
import random
from model_LMAVC import *
from dataset_LMAVC import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main(args):

    train_data = UCF101_TOG(path="../data/UCF101/***/")
    valid_data = UCF101_TOG(path="../data/UCF101/***/")
    train_data_loader = DataLoader(train_data, batch_size=args.batch, shuffle=True, num_workers=0, drop_last=True)
    valid_data_loader = DataLoader(valid_data, batch_size=args.batch, shuffle=True, num_workers=0, drop_last=True)

    loss_fn = torch.nn.MSELoss(reduction='mean').to(device)
    adam_optimizer = torch.optim.Adam(frameCodec_model.parameters(), lr=1e-4, betas=(0.9, 0.99))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(adam_optimizer, milestones=[50, 100, 150, 170], gamma=0.1)

    epochs = args.epochs  
    total_train_step = 0 

    loss = 0  
    loss_per_epoch = 0  
    loss_valid_per_epoch = 0  
    epoch_valid_count = 0 
    step_count = 0 

    for epoch in range(epochs):
        for train_data_ in train_data_loader:
            ref_OB = train_data_[0].to(device)
            frame_OB = train_data_[1].to(device)
            ref_BA = train_data_[2].to(device)
            frame_BA = train_data_[3].to(device)
            ref = ref_OB+ref_BA
            frame = frame_BA + frame_OB

            frameCodec_model.train()
            recovered_frame = frameCodec_model(ref, frame) 

            loss = loss_fn(recovered_frame, frame)  
            loss_per_epoch += loss.item()  
            step_count += 1 
            adam_optimizer.zero_grad()  
            loss.backward() 
            adam_optimizer.step()

            total_train_step += 1
            if total_train_step % 100 == 0: 
                frameCodec_model.eval()
                total_valid_loss = 0
                with torch.no_grad(): 
                    valid_count = 0
                    for valid_data_ in valid_data_loader:
                        if valid_count < 5:
                            ref_valid_OB = valid_data_[0].to(device)
                            frame_valid_OB = valid_data_[1].to(device)
                            ref_valid_BA = valid_data_[2].to(device)
                            frame_valid_BA = valid_data_[3].to(device)
                            ref_valid = ref_valid_BA + ref_valid_OB
                            frame_valid = frame_valid_BA + frame_valid_OB

                            recovered_frame_valid = frameCodec_model(ref_valid, frame_valid)
                            valid_loss = loss_fn(frame_valid, recovered_frame_valid)
                            total_valid_loss += valid_loss.item()
                            valid_count += 1
                        else:
                            break
                loss_valid_per_epoch += total_valid_loss / 5 
                epoch_valid_count += 1

        scheduler.step() 
        total_train_step = 0

    torch.save(frameCodec_model.videocompressor.state_dict(),
               "../checkpoints/C_epoch{}_loss{:.2f}.pth".format(epoch + 1, loss_per_epoch/step_count))
    torch.save(frameCodec_model.videodecompressor.state_dict(),
               "../checkpoints/DC_epoch{}_loss{:.2f}.pth".format(epoch + 1, loss_per_epoch/step_count))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', default=8, type=int)
    parser.add_argument('--epochs', default=200, type=int)

    args_ = parser.parse_args()

    main(args_)

