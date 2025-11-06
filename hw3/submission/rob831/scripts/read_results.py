import argparse
import glob
import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import torch
import matplotlib.pyplot as plt

def get_section_results(file):
    """
        requires tensorflow==1.12.0
    """
    X = []
    Y = []
    for e in tf.train.summary_iterator(file):
        for v in e.summary.value:
            if v.tag == 'Train_EnvstepsSoFar':
                X.append(v.simple_value)
            elif v.tag == 'Train_AverageReturn':
                Y.append(v.simple_value)
        if len(X) > 120:
            break
    return X, Y


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, required=True, help='path to directory contaning tensorboard results (i.e. data/q1)')
    args = parser.parse_args()

    for exp in ['*_dqn' , '*_doubledqn']:
        logdir = os.path.join(args.logdir, f"{exp}*", "events*")
        eventfiles = glob.glob(logdir)
        assert len(eventfiles), f'No event files found in {logdir}'

        result = []
        for eventfile in eventfiles:
            X, Y = get_section_results(eventfile)
            result.append(Y)

        # for i, (x, y) in enumerate(zip(X, Y)):
        #     print("Iteration {:d} | Train steps: {:d} | Return: {}".format(i, int(x), y))
        # print(len(X),len(Y))

        X.pop(0) ## trainavgreturn start from step 10,001

        # for i, (x, y) in enumerate(zip(X, Y)):
        #     print(
        #         "Iteration {:d} | Train steps: {:d} | Return: {}".format(i, int(x), y)
        #     )
        # print(len(X), len(Y))

        result = torch.Tensor(result)
        mean = result.mean(dim=0)
        std = result.std(dim=0)

        plt.errorbar(X, mean, yerr=std, label=exp.replace("*_", "").upper(), capsize=3)
        plt.xlabel("Train_EnvstepSoFar")
        plt.ylabel("Train_AverageReturn")

        plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("../../../figures/Q1.png")
    plt.tight_layout()
    print('fig saved')
