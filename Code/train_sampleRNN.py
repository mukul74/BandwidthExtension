
# Predefined imports
import os
import json
import math
import utils
import logging
import librosa
import argparse
import numpy as np
# import wandb
from platform import system
from natsort import natsorted
from datetime import datetime

import torch
import torch.nn as nn
import onnx

# Userdefined imports
from samplernn import model
from dataset import (get_dataset_filenames_split, get_dataset)
from dataset import FolderDataset, Dataloader

# torch.set_printoptions(profile="full")
from  torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
torch.initial_seed()

# wandb.init(project='Unconditional_SampleRNN_Bandwidth_Extension')
# wandb_config = wandb.config

# Default values for the model training
LOGDIR_ROOT = 'logdir' if system() == 'windows' else './logdir'
OUTDIR = './generated'
CONFIG_FILE = './default_config.json'
NUM_EPOCHS = 10
BATCH_SIZE = 10
LEARNING_RATE = 0.0003
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
VAL_FRAC = 1.0
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

    parser = argparse.ArgumentParser(description='PYTORCH UNCONDITIONAL SAMPLERNN')
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
    parser.add_argument('--target_sample_rate',                  type=int,          required=True,
                                                        help='Required for writing the predicted audio data')
    parser.add_argument('--audio_duration',             type=int,                   required=True,
                                                        help='Duration of audio file used for training')
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
    frame_sizes = config.get('frame_sizes')
    dim = config.get('dim')
    rnn_type = config.get('rnn_type')
    num_rnn_layers = config.get('num_rnn_layers')
    quantization_type = config.get('q_type')
    quantization_levels = 256 if quantization_type =='mu-law' else config.get('q_levels')
    embeding_size = config.get('emb_size')

    """Basic checks for the configuration of the model"""
    assert frame_sizes[0] < frame_sizes[1]," frame_sizes should be in increasing order"
    assert seq_len % frame_sizes[1] == 0, "Last tier should equally divide the seq_len"
    assert frame_sizes[1] % frame_sizes[0] == 0,"N-1th tier should be equally divisible by Nth Tier"

    """Calling the function to create the model and pass the required parameters"""
    """TBD SAMPLERNN MODEL"""
    model_slr = model.SampleRNN(batch_size=batch_size,frame_size=frame_sizes,q_levels=quantization_levels,
                                q_type=quantization_type,dim=dim,rnn_type=rnn_type,num_rnn_layers=num_rnn_layers,
                                seq_len=seq_len,emb_size=embeding_size,skip_connection=None,rnn_dropout=None)


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

    def data_loader(path, split_from, split_to,seq_len,overlap, q_type, q_levels,batch_size=0):
        dataset = FolderDataset(path,overlap,seq_len, q_levels, split_from, split_to)

        return Dataloader(dataset,batch_size=batch_size, seq_len=seq_len,overlap_len=overlap, shuffle=False,
                          drop_last=True)
    return data_loader

def WandB_watcher(args, config):
    pass
    # wandb_config.learning_rate = args.learning_rate
    # wandb_config.batch_size = args.batch_size
    # wandb_config.device = args.device
    # wandb_config.optimizer = args.optimizer
    # wandb_config.seq_len = config['seq_len']
    # wandb_config.frame_size_0 = config['frame_sizes'][0]
    # wandb_config.frame_size_1 = config['frame_sizes'][1]
    # wandb_config.dim = config['dim']
    # wandb_config.rnn_type = config['rnn_type']
    # wandb_config.num_rnn_layers = config['num_rnn_layers']
    # wandb_config.quantizer = config['q_type']
    # wandb_config.q_levels = config['q_levels']
    # wandb_config.emb_size = config['emb_size']


def main():

    """Reading the command line arguments for the model"""
    args = get_arguments()
    target_sample_rate = args.target_sample_rate
    input_sample_rate = args.sample_rate
    audio_duration = args.audio_duration
    num_of_iterations = (audio_duration*input_sample_rate)//1024

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

    """Pass the required arguments to WandB watcher"""
    # WandB_watcher(args,config)

    """Creating the base model and leading its parameters"""
    print("Developing the SampleRNN Model")
    net = create_model(config, args.batch_size)
    # wandb_config.model = net
    print("*"*150)
    print(net)
    print("*" * 150)

    """Declarations for the DataLoader"""
    seq_len = net.seq_len
    overlap = net.big_frame_size
    q_type = net.q_type
    q_levels = net.q_levels

    """Optimizer configuration for the model training"""
    opt = optimizer_factory[args.optimizer](net, learning_rate=args.learning_rate, momentum=args.momentum)

    """loss calculation configuration for the model"""
    criterion = nn.CrossEntropyLoss()
    # wandb_config.criterion = criterion
    """Epochs calculation, in case of resume, we need to resume from the latest checkpoint, else start form zero"""
    num_epochs = args.num_epochs
    # wandb_config.Total_epochs = num_epochs

    """Learning rate decay """
    scheduler = ReduceLROnPlateau(optimizer=opt, mode='min', patience=2, verbose=True)

    """DataLoader for the model training"""
    data_loader = get_data_loader(args)

    """Train dataset loader for input"""
    train_dataloader_input = data_loader(args.input_data_dir,0, 1-args.val_frac, seq_len, overlap, q_type=q_type,
                                         q_levels=q_levels, batch_size=args.batch_size)

    """Train dataset loader for output"""
    train_dataloader_output = data_loader(args.output_data_dir,0, 1-args.val_frac, 2*seq_len, 2*overlap, q_type=q_type,
                                          q_levels=q_levels, batch_size=args.batch_size)

    """Validation dataset loader"""
    """Currently will be used for inference hence we use the batch size to be 1"""
    validation_dataloader_input = data_loader(args.input_data_dir, 1-args.val_frac, 1, seq_len, overlap, q_type=q_type,
                                        q_levels=q_levels,batch_size=1)


    """Resume the training from the latest checkpoint or load the checkpoint for inference"""
    if args.resume is True and latest_checkpoint is not None:
        print("Checkpoint Name : ",latest_checkpoint)
        checkpoint = torch.load(latest_checkpoint, map_location='cpu')

        net.load_state_dict(checkpoint['model_state_dict'])
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        net = net.to(device)
        print(sum(p.numel() for p in net.parameters() if p.requires_grad))

        print("Epoch number at the resumed point : ", epoch)
        print("Loss at the resumed point: ",loss)
    else:
        print("Fresh Start")
        net = net.to(device)
        print(sum(p.numel() for p in net.parameters() if p.requires_grad))

    """Training Module"""
    if args.training == 'True':
        global batch_iteration_samplernn
        batch_iteration_samplernn = 0
        loss_idx_value = 0
        Tloss = 0
        for epoch in range(1, num_epochs+1):

            batch_iteration_samplernn = 0
            iteration = 0
            for iteration, (input_data, target_data) in enumerate(zip(train_dataloader_input,train_dataloader_output),iteration+1):
                loss_idx_value += 1
                hidden_reset,batch_reset = input_data[1],input_data[2] # This option resets the hidden state for a new batch of data
                if batch_reset is True:

                    print("Loading new batch of data for training")
                    batch_iteration_samplernn = 1

                batch_inputs = torch.unsqueeze(input_data[0],dim=1).to(device) # Data Format should be Batch x Channels x Samples
                batch_target = target_data[-1].to(device)

                opt.zero_grad()
                predicted_target = net((batch_inputs,hidden_reset))

                loss = criterion(predicted_target,batch_target)
                # wandb.log({"loss" : loss})

                Tloss += loss.item()

                print("Epoch {} iteration {}/{} Loss {}".format(epoch, batch_iteration_samplernn,num_of_iterations ,loss.item()))
                # if batch_iteration_samplernn%85 == 0:
                #     Tloss = Tloss/85
                #     print("Average loss for this batch : ", Tloss)
                #     Tloss = 0 # Reset the Total Loss

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
                    print("Trained model at checkpoint saved")

    else:
        strt = datetime.now()
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
            batch_inputs = torch.unsqueeze(data[0], dim=1).to(device) # Data Format should be Batch x Channels x Samp
            predicted_output = net((batch_inputs, hidden_reset))
            predicted_output = torch.max(predicted_output,1)[1]
            predicted_samples.append(predicted_output)
            # print(iteration)
            if iteration%num_of_iterations == 0:
                Generated_File_iterations_samplernn += 1
                Generated_File_name = "Generated_Unconditional_Samplernn_upsample_test_audio_new_" + \
                                      str(Generated_File_iterations_samplernn) + ".wav"
                Generated_File_path = os.path.join(args.output_dir,Generated_File_name)

                samples = torch.cat(predicted_samples,dim=1).squeeze(0)
                audio = utils.mu_law_dequantization(samples, 256)
                audio = audio.detach().cpu().numpy()
                utils.write_wav(Generated_File_path, audio, target_sample_rate)
                predicted_samples.clear()
        endt = datetime.now()
        print("Time :", endt-strt)
        print("Inference for all the files complete")


if __name__ == '__main__':
    main()