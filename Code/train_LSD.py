--input_data_dir ./data_input --output_data_dir ./data_output --num_epochs 500 --batch_size 32 --resume True --max_checkpoints 1 --checkpoint_every 25 --output_file_dur 10 --sample_rate 16000 --device cuda --training False --target_sample_rate 32000 --audio_duration 8
# Predefined imports
import os
import json
import math
from scipy import signal
import utils
import logging
import librosa
import argparse
import numpy as np
import wandb
from platform import system
from natsort import natsorted
from datetime import datetime
import librosa
import librosa.display
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import auraloss
import pandas as pd
import onnx
import librosa
# Hyperparamtere tuning parameters

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
import numpy as np
# Userdefined imports
from models import model
from dataset import (get_dataset_filenames_split, get_dataset)
from dataset import FolderDataset, Dataloader

# torch.set_printoptions(profile="full")
from  torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
torch.initial_seed()

wandb.init(project='Bandwidth Extension Using LSD as Loss function')
wandb_config = wandb.config

N, H = 2048, 1024
fftlen= "2048" #@param [256,2048, 8192]
fftlen=int(fftlen)
timesoverlap= "8"
timesoverlap=int(timesoverlap)
noverlap=fftlen -fftlen//timesoverlap
# Default values for the model training
LOGDIR_ROOT = 'logdir' if system() == 'windows' else './logdir'
OUTDIR = './generated'
CONFIG_FILE = './default_config.json'
NUM_EPOCHS = 10
BATCH_SIZE = 10
LEARNING_RATE = 0.0001
MOMENTUM = 0.9
SILENCE_THRESHOLD = None
OUTPUT_DUR = 3  # This is for the duration of generated audio data
CHECKPOINT_EVERY = 1
CHECKPOINT_POLICY = 'Always'
MAX_CHECKPOINTS = 5
RESUME = True
TRACKED_METRIC = 'val_loss'
EARLY_STOPING_PATINCE = 3
GENERATE = True
SAMPLE_RATE = 16000  # Sample rate of generated audio
SAMPLING_TEMPRATURE = [0.95]
SEED_OFFSET = 0
MAX_GENERATE_PER_EPOCH = 1
VAL_FRAC = 0.1
CLIP_GRAD = 1.0


torch.autograd.set_detect_anomaly(True)
"""Create text file for logging the all information related to training"""
if os.path.isdir(LOGDIR_ROOT + "/txt_log"):
    print("Folder exist")
else:
    os.mkdir(LOGDIR_ROOT + "/txt_log")
log_file_name = datetime.now().strftime('%d.%m.%Y_%H.%M.%S') + ".txt"
logging.basicConfig(filename=LOGDIR_ROOT + "/txt_log/" + log_file_name, filemode='a',
                    format='%(asctime)s %(msecs)d- %(process)d-%(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S %p' ,level=logging.INFO)
logging.info('Contains all the information for training')

def get_arguments():
    """
    To read all the arguments provided by the command line interface
    :return:
    Configuration for the MODEL
    """
    def check_bool(value):
        val = str(value).upper()
        if 'TRUE'.startswith(val):
            return True
        elif 'FALSE'.startswith(val):
            return False
        else:
            raise ValueError('Argument is neither `True` nor `False`')

    def check_positive(value):
        val = int(value)
        if val < 1:
            raise argparse.ArgumentTypeError("%s is not positive" % value)
        return val

    def check_max_checkpoints(value):
        if str(value).upper() != 'NONE':
            return check_positive(value)
        else:
            return None

    parser = argparse.ArgumentParser(description='Bandwidth Extension')
    parser.add_argument('--input_data_dir',             type=str,                   required=True,
                                                        help='Path to the input data directory containing the input training data')
    parser.add_argument('--output_data_dir',            type=str,                   required=True,
                                                        help='Path to the output data directory containing the output training data')
    parser.add_argument('--id',                         type=str,                   default='default', help='ID for the current training session')
    parser.add_argument('--verbose',                    type=check_bool,
                                                        help='Whether to print training step output to a new line each time (the default), or overwrite the last output', default=True)
    parser.add_argument('--batch_size',                 type=check_positive,        default=BATCH_SIZE,
                                                        help='Size of mini-batch')
    parser.add_argument('--logdir_root',                type=str,                   default=LOGDIR_ROOT,
                                                        help='Root directory for training log files')
    parser.add_argument('--config_file',                type=str,                   default=CONFIG_FILE,
                                                        help='Path to the JSON config for the model')
    parser.add_argument('--output_dir',                 type=str,                   default=OUTDIR)
    parser.add_argument('--output_file_dur',            type=check_positive,        default=OUTPUT_DUR)
    parser.add_argument('--sample_rate',                type=check_positive,        default=SAMPLE_RATE)
    parser.add_argument('--num_epochs',                 type=check_positive,        default=NUM_EPOCHS)
    parser.add_argument('--optimizer',                  type=str,                   default='adam', choices=optimizer_factory.keys(),
                                                        help='Optimizer for the training')
    parser.add_argument('--learning_rate',              type=float,                 default=LEARNING_RATE)
    parser.add_argument('--reduce_learning_rate_after', type=check_positive,
                                                        help='Exponentially reduce learning rate after this many epochs')
    parser.add_argument('--momentum',                   type=float,                 default=MOMENTUM,
                                                        help='Optimizer momentum')

    parser.add_argument('--monitor',                    type=str,                   default=TRACKED_METRIC,
                                                        choices=['loss', 'accuracy', 'val_loss', 'val_accuracy'],
                                                        help='Metric to track during training')
    parser.add_argument('--checkpoint_every',           type=check_positive,        default=CHECKPOINT_EVERY,
                                                        help='Interval (in epochs) at which to generate a checkpoint file')
    parser.add_argument('--checkpoint_policy',          type=str,                   default=CHECKPOINT_POLICY, choices=['Always', 'Best'],
                                                        help='Policy for saving checkpoints')
    parser.add_argument('--max_checkpoints',            type=check_max_checkpoints, default=MAX_CHECKPOINTS,
                                                        help='Number of checkpoints to keep on disk while training. Defaults to 5. Pass None to keep all checkpoints.')
    parser.add_argument('--resume',                     type=check_bool,            default=RESUME,
                                                        help='Whether to resume training. When True the latest checkpoint from any previous runs will be used, unless a specific checkpoint is passed using the resume_from parameter.')
    parser.add_argument('--resume_from',                type=str,
                                                        help='Checkpoint from which to resume training. Ignored when resume is False.')
    parser.add_argument('--early_stopping_patience',    type=check_positive,        default=EARLY_STOPING_PATINCE,
                                                        help='Number of epochs with no improvement after which training will be stopped.')
    parser.add_argument('--generate',                   type=check_bool,            default=GENERATE,
                                                        help='Whether to generate audio output during training. Generation is aligned with checkpoints, meaning that audio is only generated after a new checkpoint has been created.')
    parser.add_argument('--max_generate_per_epoch',     type=check_positive,        default=MAX_GENERATE_PER_EPOCH,
                                                        help='Maximum number of output files to generate at the end of each epoch')
    parser.add_argument('--temperature',                type=float,                 default=SAMPLING_TEMPRATURE, nargs='+',
                                                        help='Sampling temperature for generated audio')
    parser.add_argument('--seed',                       type=str,
                                                        help='Path to audio for seeding')
    parser.add_argument('--seed_offset',                type=int,                   default=SEED_OFFSET,
                                                        help='Starting offset of the seed audio')
    parser.add_argument('--num_val_batches',            type=int,                   default=1,
                                                        help='Number of batches to reserve for validation. DEPRECATED: This parameter now has no effect, it is retained for backward-compatibility only and will be removed in a future release. Use val_frac instead.')
    parser.add_argument('--val_frac',                   type=float,                 default=VAL_FRAC,
                                                        help='Fraction of the dataset to be set aside for validation, rounded to the nearest multiple of the batch size. Defaults to 0.1, or 10%%.')

    parser.add_argument('--device',                     type=str,                   default='cuda',
                                                        help='device for training and inference')
    parser.add_argument('--training',                   type=str,
                                                        help='Set True if you want to train the model else set it as False')
    parser.add_argument('--target_sample_rate',         type=int,                   required=True,
                                                        help='Target sample rate for processing')

    parser.add_argument('--audio_duration',             type=int,                   required=True,
                                                        help='Audio duration used for training')
    return parser.parse_args()

# initialization for the optimizer
def create_adam_optimizer(net, learning_rate, momentum):
    return torch.optim.Adam(net.parameters(),lr=learning_rate)

def create_sgd_optimizer(net, learning_rate, momentum):
    return torch.optim.SGD(net.parameters(),lr=learning_rate, momentum=momentum)

def create_rmsprop_optimizer(net, learning_rate, momentum):
    return torch.optim.RMSprop(net.parameters(),lr=learning_rate, momentum=momentum)


optimizer_factory = {'adam': create_adam_optimizer,
                     'sgd': create_sgd_optimizer,
                     'rmsprop': create_rmsprop_optimizer}


def create_model(config, batch_size):
    """
    :param config: Parameters defining the values for the neural network model
    :param batch_size: Batch Size
    :return: neural network model
    """
    seq_len = config.get('seq_len')
    overlap_len = config.get('overlap')
    kernel_size = config.get('kernel_size')


    """Calling the function to create the model and pass the required parameters"""

    model_slr = model.BWExtender(seq_len,overlap_len, kernel_size=kernel_size)


    return model_slr


def get_latest_checkpoint(logdir):
    """
    :param logdir: Folder for saving and loading the checkpoint
    :return: latest checkpoint
    """
    rundir_datetimes = []
    try:
        for f in os.listdir(logdir):
            if os.path.isdir(os.path.join(logdir, f)):
                dt = datetime.strptime(f, '%d.%m.%Y_%H.%M.%S')
                rundir_datetimes.append(dt)
    except ValueError as err:
        print(err)
    if len(rundir_datetimes) > 0:
        i = 0
        rundir_datetimes = natsorted(rundir_datetimes, reverse=True)
        latest_checkpoint = None
        while (latest_checkpoint == None):
            ltst_checkpoint = []
            rundir = rundir_datetimes[0].strftime('%d.%m.%Y_%H.%M.%S')
            for cpts in os.listdir(os.path.join(logdir,rundir)):
                ltst_checkpoint.append(cpts)

            ltst_checkpoint = natsorted(ltst_checkpoint, reverse=True)
            ltst_checkpoint_path = os.path.join(logdir,rundir,ltst_checkpoint[0])
            latest_checkpoint = ltst_checkpoint_path
        return latest_checkpoint

def get_data_loader(args):

    def data_loader(path, split_from, split_to,overlap_len,frame_len,batch_size=0):
        dataset = FolderDataset(path,frame_len,overlap_len,split_from, split_to)

        return Dataloader(dataset,batch_size=batch_size,seq_len=frame_len, overlap_len=overlap_len,shuffle=False,
                          drop_last=True)
    return data_loader

def WandB_watcher(args, config):
    wandb_config.learning_rate = args.learning_rate
    wandb_config.batch_size = args.batch_size
    wandb_config.device = args.device
    wandb_config.optimizer = args.optimizer
    wandb_config.seq_len = config['seq_len']
    wandb_config.frame_size_0 = config['frame_sizes'][0]
    wandb_config.frame_size_1 = config['frame_sizes'][1]
    wandb_config.dim = config['dim']
    wandb_config.rnn_type = config['rnn_type']
    wandb_config.num_rnn_layers = config['num_rnn_layers']
    wandb_config.quantizer = config['q_type']
    wandb_config.q_levels = config['q_levels']
    wandb_config.emb_size = config['emb_size']


def LSD_mse_loss(orig,rec,criterion):
    loss = criterion(orig,rec)
    loss = (torch.sqrt((loss**2).mean(dim=1))).mean(dim=1).mean(dim=0)
    return loss


def main():

    """Reading the command line arguments for the model"""
    args = get_arguments()


    """Cuda information"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device Available : ', device)
    device = args.device
    print('Device selected for Training/Inference : ', device)

    if args.training == 'True':
        print("Model is in training mode now")
    else:
        print("Model is in inference mode now")

    """Creating the directory to log the session"""
    logdir = os.path.join(args.logdir_root, args.id)
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    """ Directory for generated data"""
    generate_dir = os.path.join(args.output_dir, args.id)
    if not os.path.exists(generate_dir):
        os.makedirs(generate_dir)

    rundir = '{}/{}'.format(logdir, datetime.now().strftime('%d.%m.%Y_%H.%M.%S'))

    """Reading the latest checkpoint for resuming training or inference"""
    latest_checkpoint = get_latest_checkpoint(logdir)

    """Loading the model configuration"""
    print("Loading the configurations for the model from the json file")
    with open(args.config_file, 'r') as config_file:
        config = json.load(config_file)

    audio_duration = args.audio_duration
    target_samplerate = args.target_sample_rate
    input_samplerate = args.sample_rate
    blocks = (audio_duration*input_samplerate)//(config['seq_len'] - config['overlap'])

    """Pass the required arguments to WandB watcher"""
    WandB_watcher(args,config)

    """Creating the base model and leading its parameters"""
    print("Developing the SampleRNN Model")
    net = create_model(config, args.batch_size)
    wandb_config.model = net
    print("*"*150)
    print(net)
    print("*" * 150)

    """Declarations for the DataLoader"""
    frame_size = net.frame_size
    overlap_len = net.overlap_len


    """Optimizer configuration for the model training"""
    opt = optimizer_factory[args.optimizer](net, learning_rate=args.learning_rate, momentum=args.momentum)

    """Epochs calculation, in case of resume, we need to resume from the latest checkpoint, else start form zero"""
    num_epochs = args.num_epochs
    wandb_config.Total_epochs = num_epochs


    """DataLoader for the model training"""
    data_loader = get_data_loader(args)

    """Train dataset loader for input"""
    train_dataloader_input = data_loader(args.input_data_dir,0, 1-args.val_frac,overlap_len,frame_size, batch_size=args.batch_size)
    # train_dataloader_input = data_loader(args.input_data_dir,0, 1-args.val_frac,overlap_len,frame_size, batch_size=1)
    """Train dataset loader for output"""
    train_dataloader_output = data_loader(args.output_data_dir,0, 1-args.val_frac,2*overlap_len,2*frame_size, batch_size=args.batch_size)

    """Validation dataset loader"""
    """Currently will be used for inference hence we use the batch size to be 1"""
    validation_dataloader_input = data_loader(args.input_data_dir, 1-args.val_frac,1,overlap_len,frame_size,batch_size=1)



    """Resume the training from the latest checkpoint or load the checkpoint for inference"""
    if args.resume is True and latest_checkpoint is not None:
        print("Checkpoint Name : ",latest_checkpoint)
        checkpoint = torch.load(latest_checkpoint, map_location='cpu')

        net.load_state_dict(checkpoint['model_state_dict'])
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        net = net.to(device)

        print("Epoch number at the resumed point : ", epoch)
        print("Loss at the resumed point: ",loss)
    else:
        print("Fresh Start")
        net = net.to(device)

    """Training Module"""
    if args.training == 'True':
        global batch_iteration_samplernn
        batch_iteration_samplernn = 1
        loss_idx_value = 0
        Tloss = 0
        for epoch in range(1, num_epochs+1):

            batch_iteration_samplernn = 1
            iteration = 0
            for iteration, (input_data, target_data) in enumerate(zip(train_dataloader_input,train_dataloader_output),iteration+1):
                loss_idx_value += 1
                hidden_reset,batch_reset = input_data[1],input_data[2] # This option resets the hidden state for a new batch of data
                if batch_reset is True:

                    print("Loading new batch of data for training")
                    batch_iteration_samplernn = 1

                batch_inputs = torch.unsqueeze(input_data[0],dim=1).to(device) # Data Format should be Batch x Channels x Samples
                batch_target = torch.unsqueeze(target_data[-1],dim=1).to(device)


                opt.zero_grad()
                predicted_target = net((batch_inputs,hidden_reset))

                STFT_Original = torch.stft(batch_target.squeeze(), n_fft=1024, return_complex=True)
                STFT_Reconstructed = torch.stft(predicted_target.squeeze(), n_fft=1024, return_complex=True)
                loss = ((torch.sqrt(((torch.log10(torch.abs(STFT_Original)**2 + 1e-7) - torch.log10(torch.abs(STFT_Reconstructed)**2 + 1e-7))**2).mean(dim=1))).mean(dim=1)).mean(dim=0)
                Tloss += loss.item()

                print("Epoch {} Iteration {}/{} Loss {}".format(epoch, batch_iteration_samplernn, blocks,loss.item()))
                if batch_iteration_samplernn%blocks == 0:
                    Tloss = Tloss/blocks
                    print("Epoch {} Average loss for this batch while training {}".format(epoch, Tloss))
                    Tloss = 0 # Reset the Total Loss

                writer.add_scalar("Loss",loss.item(),loss_idx_value)
                batch_iteration_samplernn += 1

                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), CLIP_GRAD)
                opt.step()

            if epoch % args.checkpoint_every == 0: # TO save the check points

                    print("Saving the checkpoint at epoch ", epoch)

                    if not os.path.exists(rundir):
                        os.makedirs(rundir)

                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': net.state_dict(),
                        'optimizer_state_dict': opt.state_dict(),
                        'loss': Tloss,
                    }, rundir +'/' +'Training_checkpoint_at_epoch_' + str(epoch) + '.pth')
                    print("Trained model saved at this particular checkpoint")


    else:
        print("Inference mode running")
        predicted_samples = []
        "Evaluation Mode"
        net.eval()
        iteration = 0
        print("Total Number of Files in Validation : ",len(validation_dataloader_input.dataset.file_names))
        Generated_File_iterations_samplernn = 0
        for (iteration, data) in enumerate(validation_dataloader_input, iteration + 1):
            hidden_reset, batch_reset = data[1], data[2]  # This option resets the hidden state for a new batch of data
            if batch_reset is True:
                print("Loading new batch of data for Inference")
                batch_iteration_samplernn = 0
            batch_inputs = torch.unsqueeze(data[0], dim=1).to(device) # Data Format should be Batch x Channels x Sample
            predicted_output = net((batch_inputs, hidden_reset))
            predicted_output = torch.squeeze(torch.squeeze(predicted_output, dim=0), dim=0)
            predicted_samples.append(predicted_output)
            if iteration%blocks == 0:
                Generated_File_iterations_samplernn += 1
                Generated_File_name = "Generated_audio_files_" + \
                                      str(Generated_File_iterations_samplernn) + ".wav"
                Generated_File_path = os.path.join(args.output_dir,Generated_File_name)


                samples = torch.cat(predicted_samples)
                audio = samples.detach().cpu().numpy()
                utils.write_wav(Generated_File_path, audio, target_samplerate)


                predicted_samples.clear()

        print("Inference for all the files complete")

if __name__ == '__main__':
    main()