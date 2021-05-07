"""
Run Command:
python -m train_lstm --env_name fruitbot
"""

""" USE CPU """
# import tensorflow as tf
""" USE GPU """
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# from baselines.ppo2 import ppo2
from baselines.a2c import a2c
# from baselines.common.models import build_impala_cnn
from baselines.common.models import cnn_lstm, lstm, cnn_lnlstm
from baselines.common.mpi_util import setup_mpi_gpus
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

def train_fn(env_name, num_envs, distribution_mode, num_levels, start_level, timesteps_per_proc, is_test_worker=False, log_dir='/tmp/procgen', comm=None):
    learning_rate = 5e-4
    vf_coef = 0.5
    ent_coef = .01
    gamma = .999
    lam = .95
    nsteps = 10000
    nminibatches = 8
    log_interval = 1000
    ppo_epochs = 3
    clip_range = .2
    use_vf_clipping = True
    timesteps_per_proc = 10000

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

    # conv_fn = lambda x: build_impala_cnn(x, depths=[16,32,32], emb_size=256)
    # LSTM CSNN
    network = lstm(nlstm = 80)
    # LSTM Layer Normalized CNN
    # network = lambda x: CnnLnLstmPolicy(x, depths=[16,32,32], emb_size=256)


    logger.info("training")

    # ppo2.learn(
    #     env=venv,
    #     network=conv_fn,
    #     total_timesteps=timesteps_per_proc,
    #     save_interval=0,
    #     nsteps=nsteps,
    #     nminibatches=nminibatches,
    #     lam=lam,
    #     gamma=gamma,
    #     noptepochs=ppo_epochs,
    #     log_interval=1,
    #     ent_coef=ent_coef,
    #     mpi_rank_weight=mpi_rank_weight,
    #     clip_vf=use_vf_clipping,
    #     comm=comm,
    #     lr=learning_rate,
    #     cliprange=clip_range,
    #     update_fn=None,
    #     init_fn=None,
    #     vf_coef=0.5,
    #     max_grad_norm=0.5,
    # )
    a2c.learn(
        network = network, #TODO
        env = venv,
        nsteps = nsteps,
        total_timesteps = timesteps_per_proc,
        vf_coef = vf_coef, 
        log_interval = log_interval, # how often to log results
        ent_coef = ent_coef,
        seed = 1
    )

# def build_lstm(unscaled_imgs, nlstm):


def main():
    parser = argparse.ArgumentParser(description='Process procgen training arguments.')
    parser.add_argument('--env_name', type=str, default='coinrun')
    parser.add_argument('--num_envs', type=int, default=64)
    parser.add_argument('--distribution_mode', type=str, default='hard', choices=["easy", "hard", "exploration", "memory", "extreme"])
    parser.add_argument('--num_levels', type=int, default=0)
    parser.add_argument('--start_level', type=int, default=0)
    parser.add_argument('--test_worker_interval', type=int, default=0)
    parser.add_argument('--timesteps_per_proc', type=int, default=50_000_000)

    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    is_test_worker = False
    test_worker_interval = args.test_worker_interval

    if test_worker_interval > 0:
        is_test_worker = rank % test_worker_interval == (test_worker_interval - 1)

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