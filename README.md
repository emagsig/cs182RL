Requirements: MuJoCo 1.6

GPU Related:
Nvidia Cuda Stuff
Tensorflow Stuff
---------------------------------------------------

## Install

#### Initial Steps:

1) Install Mujoco 150: https://www.roboti.us/
##### NOTE: When we installed Mujoco 1.5, since Mujoco 2.0 already existed as our default Mujoco PATH in Windows (from HW), we had to move "glfw3.dll", "glfw3.lib", and "glfw3static.lib" from "mjpro150/bin" to "mujoco200/bin"

2) To use GPU: Install necessary CUDA/cuDNN dependencies by following directions in this link: https://www.tensorflow.org/install/gpu
(If you don't want to use GPU, ignore this step and follow instruction 5) below)

3) Install Anaconda 

4) Install env dependencies with cmd line 
```
conda env update --name 182RLproj --file environment.yml
conda activate 182RLproj
pip install https://github.com/openai/baselines/archive/9ee399f5b20cd70ac0a871927a6cf043b478193f.zip
```

5) Optional: If you would like to use CPU for training instead of GPU, install "tensorflow == 1.15" into our conda environment


## Training
Choose from the below arguments to fill in [CNN TYPE], [LSTM TYPE], [Boolean]

Choices for arguments:
> cnn_choices = ["none", "base_impala", "leaky_sigmoid_impala", "sigmoid_leaky_impala", "leaky_relu_impala", "sigmoid_impala", "absolute_relu_impala"]
> lstm_choices = ["none", "base_lstm", "cnn_lstm"]
> layer_norm_choices = [True, False]

Run the following commands in cmd line after activating the conda environment.

#### Training 1M Runs:
FORMAT:
```
python -m train_procgen.train --env_name fruitbot --num_levels 50 --timesteps_per_proc 1_000_000 --start_level 500 --num_envs 32 --cnn_type [CNN TYPE] --lstm_type [LSTM TYPE] --layer_norm [Boolean]
```
EXAMPLE:
```
python -m train_procgen.train --env_name fruitbot --num_levels 50 --timesteps_per_proc 1_000_000 --start_level 500 --num_envs 32 --cnn_type leaky_relu_impala --lstm_type cnn_lstm --layer_norm True
```

#### Training 50M Runs:
FORMAT:
```
python -m train_procgen.train --env_name fruitbot --num_levels 100 --timesteps_per_proc 50_000_000 --start_level 500 --num_envs 32 --cnn_type [CNN TYPE] --lstm_type [LSTM TYPE] --layer_norm [Boolean]
```
EXAMPLE:
```
python -m train_procgen.train --env_name fruitbot --num_levels 100 --timesteps_per_proc 50_000_000 --start_level 500 --num_envs 32 --cnn_type absolute_relu_impala --lstm_type none --layer_norm True
```

Find training results in folder "train_results\".

##### NOTE: When training, model checkpoints are saved to "train_results\checkpoints\XXXXX" and training logs are saved to "train_results\progress.csv". Rename folder and logs file to make sure that training results are not overrun when running another training session

## Testing
FORMAT:
```
python -m train_procgen.train --test True --num_levels 500 --start_level 0 --timesteps_per_proc 20000 --load_path train_results\[Model Checkpoint Directory]\[Checkpoint Number]
```
EXAMPLE:
```
python -m train_procgen.train --test True --num_levels 500 --start_level 50 --timesteps_per_proc 20000 --load_path train_results\50M_absrelu_checkpoints\05020
python -m train_procgen.train --test True --num_levels 500 --start_level 0 --timesteps_per_proc 20000 --load_path train_results\checkpoints\00060
```

Find testing results in folder "test_results\progress.csv". 
"eplenmean": Mean of episode lengths from testing
"eprewmean": Mean of episode rewards from training
