import argparse
import numpy as np
import random
import torch
import torch.nn as nn

from modules import *

def main(args):

    if args.game == "Alien":
        env_id = "Alien-v0"
        cnn = True
    elif args.game == "Cartpole":
        env_id = "CartPole-v0"
        cnn = False
    else:
        raise ValueError("Unsupported Game")

    env, val_env = get_env(env_id)
    results = train(env, val_env,
                    args.method,
                    args.variance, args.mean,
                    args.decision_eps,
                    args.alpha, args.beta,
                    args.hardcoded,
                    cnn,
                    num_trials=args.num_trials)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Robust Replay Buffer Experiments")

    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--beta", type=float, default=0.4)
    parser.add_argument("--decision_eps", type=float, default=0.)
    parser.add_argument("--hardcoded", type=bool, default=False)
    parser.add_argument("--mean", type=float, default=0.)
    parser.add_argument("--method", type=str, choices=['PER', 'average_over_batch', 'average_over_all'])
    parser.add_argument("--num_trials", type=int, default=10)
    parser.add_argument("--variance", type=float, default=1.)

    parser.add_argument("--game", type=str, choices=["Cartpole", "Alien"], default="Cartpole")

    args = parser.parse_args()

    main(args)
