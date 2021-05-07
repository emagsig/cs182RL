"""
Run:
python -m train_procgen.train --env_name fruitbot --num_levels 50 --timesteps_per_proc 1_000_000 --start_level 500 --num_envs 32 --test_worker_interval 4
"""

""" USE CPU """
# import tensorflow as tf
""" USE GPU """
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from baselines.ppo2 import ppo2

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

def train_fn(env_name, num_envs, distribution_mode, num_levels, start_level, timesteps_per_proc, is_test_worker=False, log_dir='/tmp/procgen', comm=None):
    learning_rate = 5e-4
    ent_coef = .01
    gamma = .999
    lam = .95
    nsteps = 256
    nminibatches = 8
    ppo_epochs = 3
    clip_range = .2
    use_vf_clipping = True
    log_interval = 10
    save_interval = 10 # default 0
    # timesteps_per_proc = timesteps_per_proc

    mpi_rank_weight = 0 if is_test_worker else 1
    num_levels = 0 if is_test_worker else num_levels

    log_dir = './train_results'
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
    # print("base_impala_model")
    # conv_fn = lambda x: base_impala_model(x, depths=[16,32,32,32])

    # print("leaky_sigmoid_impala_model")
    # conv_fn = lambda x: leaky_sigmoid_impala_model(x, depths=[16,32,32,32])

    # print("sigmoid_leaky_impala_model")
    # conv_fn = lambda x: sigmoid_leaky_impala_model(x, depths=[16,32,32,32])

    # print("leaky_relu_impala_model")
    # conv_fn = lambda x: leaky_relu_impala_model(x, depths=[16,32,32,32])

    # print("sigmoid_impala_model")
    # conv_fn = lambda x: sigmoid_impala_model(x, depths=[16,32,32,32])

    # print("build_impala_cnn")
    # conv_fn = lambda x: build_impala_cnn(x, depths=[16,32,32], emb_size=256)

    # print("absolute_relu_impala_model")
    # conv_fn = lambda x: absolute_relu_impala_model(x, depths=[16,32,32,32])

    '''
    LSTM
    '''
    # print("lstm_base")
    # conv_fn = lstm_base(nlstm=128, layer_norm=False)

    # print("lstmbase_lnorm_")
    # conv_fn = lstm_base(nlstm=128, layer_norm=True)

    # print("lstmcnn_impala_")
    # conv_fn = lstm_cnn(nlstm=128, layer_norm=True, conv_fn=base_impala_model, depths = [16,32,32,32]) # **kwargs for cnn passed in normally

    # print("lstmcnn_leakysigmoid_")
    # conv_fn = lstm_cnn(nlstm=128, layer_norm=True, conv_fn=leaky_sigmoid_impala_model, depths = [16,32,32,32]) # **kwargs for cnn passed in normally

    # print("lstmcnn_sigmoidleaky_")


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
    )

def main():
    parser = argparse.ArgumentParser(description='Process procgen training arguments.')
    parser.add_argument('--env_name', type=str, default='coinrun')
    parser.add_argument('--num_envs', type=int, default=64)
    parser.add_argument('--distribution_mode', type=str, default='hard', choices=["easy", "hard", "exploration", "memory", "extreme"])
    parser.add_argument('--num_levels', type=int, default=0)
    parser.add_argument('--start_level', type=int, default=0)
    parser.add_argument('--test_worker_interval', type=int, default=0)
    parser.add_argument('--timesteps_per_proc', type=int, default=50000000)

    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    is_test_worker = False
    test_worker_interval = args.test_worker_interval

    if test_worker_interval > 0:
        is_test_worker = rank % test_worker_interval == (test_worker_interval - 1)
        #print("hi")

    train_fn(args.env_name,
        args.num_envs,
        args.distribution_mode,
        args.num_levels,
        args.start_level,
        args.timesteps_per_proc,
        is_test_worker=is_test_worker,
        comm=comm)

if __name__ == '__main__':
    main()
    print("All Done")