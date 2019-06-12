from modules import *

env, val_env = get_env("CartPole-v0")
result = train(env, val_env, 0, 1, False, False, False, False)
