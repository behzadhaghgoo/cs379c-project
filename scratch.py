from modules import *

env, val_env = get_env("CartPole-v0")
result = train(env, val_env, "PER", 1., 0., 0.5, 0.6, 0.4, False)
