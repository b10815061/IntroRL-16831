
# Colab
I ran the code in a Colab environment. Unfortunately, I was unable to set up the environment on my Mac, so I apologize for not testing the Python command line locally.

## Behavioral cloning (part 2)
```
python rob831/scripts/run_hw1.py \
--expert_policy_file rob831/policies/experts/Ant.pkl \
--env_name Ant-v2 --exp_name bc_Ant_p2 --n_iter 1 \
--expert_data rob831/expert_data/expert_data_Ant-v2.pkl \
--n_layers 2 --video_log_freq -1 \
--eval_batch_size 2000
```

```
python rob831/scripts/run_hw1.py \
--expert_policy_file rob831/policies/experts/HalfCheetah.pkl \
--env_name HalfCheetah-v2 --exp_name bc_HalfCheetah_p2 --n_iter 1 \
--expert_data rob831/expert_data/expert_data_HalfCheetah-v2.pkl \
--n_layers 2 --video_log_freq -1 \
--eval_batch_size 2000
```

```
python rob831/scripts/run_hw1.py \
--expert_policy_file rob831/policies/experts/Hopper.pkl \
--env_name Hopper-v2 --exp_name bc_Hopper_p2 --n_iter 1 \
--expert_data rob831/expert_data/expert_data_Hopper-v2.pkl \
--n_layers 2 --video_log_freq -1 \
--eval_batch_size 2000
```

```
python rob831/scripts/run_hw1.py \
--expert_policy_file rob831/policies/experts/Humanoid.pkl \
--env_name Humanoid-v2 --exp_name bc_Humanoid_p2 --n_iter 1 \
--expert_data rob831/expert_data/expert_data_Humanoid-v2.pkl \
--n_layers 2 --video_log_freq -1 \
--eval_batch_size 2000
```

```
python rob831/scripts/run_hw1.py \
--expert_policy_file rob831/policies/experts/Walker2d.pkl \
--env_name Walker2d-v2 --exp_name bc_Walker2d_p2 --n_iter 1 \
--expert_data rob831/expert_data/expert_data_Walker2d-v2.pkl \
--n_layers 2 --video_log_freq -1 \
--eval_batch_size 2000
```

## DAgger

```
python rob831/scripts/run_hw1.py \
--expert_policy_file rob831/policies/experts/Ant.pkl \
--env_name Ant-v2 --exp_name q2_dagger_Ant --n_iter 8 --do_dagger \
--expert_data rob831/expert_data/expert_data_Ant-v2.pkl \ 
--n_layers 5 --video_log_freq -1 --learning_rate 7.6e-3 \
--eval_batch_size 5000
```

```
python rob831/scripts/run_hw1.py \
--expert_policy_file rob831/policies/experts/Hopper.pkl \
--env_name Hopper-v2 --exp_name q2_dagger_Hopper --n_iter 8 --do_dagger \
--expert_data rob831/expert_data/expert_data_Hopper-v2.pkl \
--n_layers 5 --video_log_freq -1 --learning_rate 7.6e-3 \
--eval_batch_size 5000
```