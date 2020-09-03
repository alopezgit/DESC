import time
import sys
from options.train_options import TrainOptions
from data import create_dataloader
from models import create_model
from utils.util import SaveResults
import numpy as np
import torch
import cv2

cv2.setNumThreads(0)
torch.manual_seed(0)
torch.backends.cudnn.benchmark = True
np.random.seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)

if __name__ == '__main__':
    # Initialize training options and dataloader
    opt = TrainOptions().parse()
    train_data_loader = create_dataloader(opt)
    n_iters_epoch = len(train_data_loader)
    max_epoch = opt.n_train_epochs
    print('#training images = %d' % n_iters_epoch)
    finished_training = 0

    # Initialize model
    model = create_model(opt)
    model.setup(opt)
    model.total_images = 0
    model.total_steps = 0
    model.experiment_name = opt.experiment_name

    save_results = SaveResults(opt)

    # If in the second step (joint training) load pretrained models
    if opt.load_pretrained:
        model.load_networks('latest_pretrain')

    for epoch in range(opt.epoch_count, max_epoch+1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        model.epoch = epoch
        # Epoch loop
        print("training stage (epoch: %s) starting...................." % epoch)
        for ind, data in enumerate(train_data_loader):
            iter_start_time = time.time()
            if model.total_images % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            model.total_images += opt.batchSize
            model.total_steps += 1
            epoch_iter += 1
            model.set_input(data)
            model.optimize_parameters()
            if model.total_images % (opt.batchSize*int(opt.print_freq/opt.batchSize)) == 0:
                # These lines prints losses
                losses = model.get_current_losses()
                t = (time.time() - iter_start_time)
                save_results.print_current_losses(
                    epoch, epoch_iter, n_iters_epoch, model.total_steps, losses, t, t_data)

            iter_data_time = time.time()
            if model.total_steps == opt.n_train_iterations:
                # In the second step we stop after N training iterations
                print('Finished training')
                finished_training = 1
                break
        if finished_training:
            break
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, model.total_steps))
            model.save_networks(str(epoch)+'_'+model.experiment_name)
        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, max_epoch, time.time() - epoch_start_time))
    # Save final model
    model.save_networks('latest_'+model.experiment_name)
