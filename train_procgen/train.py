# TODO: See for testing https://github.com/ana-tudor/classyconditioning/

"""
COMMAND LINE PROMPTS

Choices for arguments:
cnn_choices = ["none", "base_impala", "leaky_sigmoid_impala", "sigmoid_leaky_impala", "leaky_relu_impala", "sigmoid_impala", "absolute_relu_impala"]
lstm_choices = ["none", "base_lstm", "cnn_lstm"]
layer_norm_choices = [True, False]

Training 1M Run:
FORMAT:
python -m train_procgen.train --env_name fruitbot --num_levels 50 --timesteps_per_proc 1_000_000 --start_level 500 --num_envs 32 --cnn_type [CNN TYPE] --lstm_type [LSTM TYPE] --layer_norm [Boolean]
EXAMPLE:
python -m train_procgen.train --env_name fruitbot --num_levels 50 --timesteps_per_proc 1_000_000 --start_level 500 --num_envs 32 --cnn_type leaky_relu_impala --lstm_type cnn_lstm --layer_norm True

Training 50M Runs:
FORMAT:
python -m train_procgen.train --env_name fruitbot --num_levels 100 --timesteps_per_proc 50_000_000 --start_level 500 --num_envs 32 --cnn_type [CNN TYPE] --lstm_type [LSTM TYPE] --layer_norm [Boolean]
EXAMPLE:
python -m train_procgen.train --env_name fruitbot --num_levels 100 --timesteps_per_proc 50_000_000 --start_level 500 --num_envs 32 --cnn_type absolute_relu_impala --lstm_type none --layer_norm True

Testing:
FORMAT:
python -m train_procgen.train --test True --num_levels 500 --start_level 0 --timesteps_per_proc 20000 --load_path train_results\[Model Checkpoint Directory]\[Checkpoint Number]
EXAMPLE:
python -m train_procgen.train --test True --num_levels 500 --start_level 0 --timesteps_per_proc 100000 --load_path train_results\50M_absrelu_checkpoints\05020
python -m train_procgen.train --test True --num_levels 500 --start_level 0 --timesteps_per_proc 100000 --load_path train_results\50M_lstmcnn_leakyrelu_checkpoints\04760
"""

# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# from baselines.ppo2 import ppo2
from . import ppo2

from .models.base_impala import base_impala_model
from .models.sigmoid_impala import sigmoid_impala_model
from .models.leaky_relu_impala import leaky_relu_impala_model
from .models.sigmoid_leaky_impala import sigmoid_leaky_impala_model
from .models.leaky_sigmoid_impala import leaky_sigmoid_impala_model
from .models.absolute_relu_model import absolute_relu_impala_model

from baselines.common.mpi_util import setup_mpi_gpus
from baselines.common.models import build_impala_cnn
from procgen import ProcgenEnv
from baselines.common.vec_env import (
    VecExtractDictObs,
    VecMonitor,
    VecFrameStack,
    VecNormalize
)
from baselines import logger
from mpi4py import MPI
import argparse
import os

from .models.base_impala import base_impala_model
from .models.sigmoid_impala import sigmoid_impala_model
from .models.leaky_relu_impala import leaky_relu_impala_model
from .models.sigmoid_leaky_impala import sigmoid_leaky_impala_model
from .models.leaky_sigmoid_impala import leaky_sigmoid_impala_model
from .models.absolute_relu import absolute_relu_impala_model

from .models.lstm_base import lstm_base
from .models.lstmcnn_base import lstm_cnn

def train_fn(env_name, num_envs, distribution_mode, num_levels, start_level, 
    timesteps_per_proc, cnn_type, lstm_type, layer_norm, load_path, test, is_test_worker=False, log_dir='/tmp/procgen', comm=None):
    
    learning_rate = 5e-4
    ent_coef = .01
    gamma = .999
    lam = .95
    nsteps = 256
    nminibatches = 8
    ppo_epochs = 3
    clip_range = .2
    use_vf_clipping = True
    log_interval = 1
    save_interval = 1 # default 0
    # timesteps_per_proc = timesteps_per_proc

    mpi_rank_weight = 0 if is_test_worker else 1
    num_levels = 0 if is_test_worker else num_levels

    log_dir = './train_results'
    if test:
        log_dir = './test_results'
    print("Log dir: ", log_dir)
    os.path.exists(log_dir)

    if log_dir is not None:
        log_comm = comm.Split(1 if is_test_worker else 0, 0)
        format_strs = ['csv', 'stdout'] if log_comm.Get_rank() == 0 else []
        logger.configure(comm=log_comm, dir=log_dir, format_strs=format_strs)

    logger.info("creating environment")
    venv = ProcgenEnv(num_envs=num_envs, env_name=env_name, num_levels=num_levels, start_level=start_level, distribution_mode=distribution_mode)
    venv = VecExtractDictObs(venv, "rgb")

    venv = VecMonitor(
        venv=venv, filename=None, keep_buf=100,
    )

    venv = VecNormalize(venv=venv, ob=False)

    logger.info("creating tf session")
    setup_mpi_gpus()
    config = tf.ConfigProto()
    # config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True #pylint: disable=E1101
    sess = tf.Session(config=config)
    sess.__enter__()

    logger.info("training")


    '''
    IMPALA
    '''
    cnn_choices = {"none": None, "base_impala": base_impala_model, "leaky_sigmoid_impala": leaky_sigmoid_impala_model, 
        "sigmoid_leaky_impala": sigmoid_leaky_impala_model, "leaky_relu_impala": leaky_relu_impala_model, 
        "sigmoid_impala": sigmoid_impala_model, "absolute_relu_impala": absolute_relu_impala_model}
    cnn = cnn_choices[cnn_type]

    conv_fn = None
    if cnn != None:
        conv_fn = lambda x: cnn(x, depths=[16,32,32,32])

    '''
    LSTM
    '''
    if (cnn != None) and (lstm_type == "cnn_lstm"):
        conv_fn = lstm_cnn(nlstm=256, layer_norm=layer_norm, conv_fn=cnn, depths = [16,32,32,32])
        # conv_fn = lstm_cnn(nlstm=128, layer_norm=layer_norm, conv_fn=cnn, depths = [16,32,32,32])
    elif lstm_type == "base_lstm":
        conv_fn = lstm_base(nlstm=256, layer_norm=layer_norm)

    if conv_fn == None:
        raise ValueError("conv_fn CANNOT be None")


    logger.info("training")
    ppo2.learn(
        env=venv,
        network=conv_fn,
        total_timesteps=timesteps_per_proc,
        save_interval=save_interval,
        nsteps=nsteps,
        nminibatches=nminibatches,
        lam=lam,
        gamma=gamma,
        noptepochs=ppo_epochs,
        log_interval=log_interval,
        ent_coef=ent_coef,
        mpi_rank_weight=mpi_rank_weight,
        clip_vf=use_vf_clipping,
        comm=comm,
        lr=learning_rate,
        cliprange=clip_range,
        update_fn=None,
        init_fn=None,
        vf_coef=0.5,
        max_grad_norm=0.5,
        load_path = load_path,
        test = test
    )

def main():
    cnn_choices = ["none", "base_impala", "leaky_sigmoid_impala", "sigmoid_leaky_impala", "leaky_relu_impala", "sigmoid_impala", "absolute_relu_impala"]
    lstm_choices = ["none", "base_lstm", "cnn_lstm"]

    parser = argparse.ArgumentParser(description='Process procgen training arguments.')
    parser.add_argument('--env_name', type=str, default='coinrun')
    parser.add_argument('--num_envs', type=int, default=64)
    parser.add_argument('--distribution_mode', type=str, default='hard', choices=["easy", "hard", "exploration", "memory", "extreme"])
    parser.add_argument('--num_levels', type=int, default=0)
    parser.add_argument('--start_level', type=int, default=0)
    parser.add_argument('--test_worker_interval', type=int, default=0)
    parser.add_argument('--timesteps_per_proc', type=int, default=1000000)
    # For testing
    parser.add_argument('--load_path', type=str, default=None,
        help='The relative or absolute path to a model checkpoint if an initial \
                load from this checkpoint is desired')
    parser.add_argument('--test', type=bool, default=False,
        help='True if the model should run as a testing agent, and should not be updated')
    # choose model type
    parser.add_argument('--cnn_type', type=str, default='base_impala', choices=cnn_choices,
        help='Choose the type of cnn to train: \
            [none, base_impala, leaky_sigmoid_impala, sigmoid_leaky_impala, leaky_relu_impala, sigmoid_impala, absolute_relu_impala]')
    parser.add_argument('--lstm_type', type=str, default='none', choices=lstm_choices,
        help='Choose the type of lstm to train: \
            [none, base_lstm, cnn_lstm]')
    parser.add_argument('--layer_norm', type=bool, default=True,
        help='Choose whether the lstm should have layer normalization: \
            [True, False]')

    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    is_test_worker = False
    test_worker_interval = args.test_worker_interval

    if test_worker_interval > 0:
        is_test_worker = rank % test_worker_interval == (test_worker_interval - 1)
        #print("hi")
    
    if args.lstm_type == "base_lstm" and args.cnn_type != "none":
        args.cnn_type = "none"
        print("MODEL CHOICE ERROR: The 'base_lstm' model does not implement CNNs so the CNN choice will be disregarded, choose 'cnn_lstm' if you would like to combine the LSTM and CNN")
    elif args.cnn_type == "none" and args.lstm_type != "base_lstm":
        args.cnn_type = "base_lstm"
        print("MODEL CHOICE WARNING: Cannot train a CNN-LSTM model without a CNN model choice selected. Training will default to the 'base_lstm' CNN model")

    print("MODEL CONFIGURATION: CNN Type: ", args.cnn_type, " LSTM type: ", args.lstm_type, " Layer Normalization: ", args.layer_norm)

    train_fn(args.env_name,
        args.num_envs,
        args.distribution_mode,
        args.num_levels,
        args.start_level,
        args.timesteps_per_proc,
        args.cnn_type,
        args.lstm_type,
        args.layer_norm,
        args.load_path,
        args.test,
        is_test_worker=is_test_worker,
        comm=comm)

if __name__ == '__main__':
    main()
    print("All Done")