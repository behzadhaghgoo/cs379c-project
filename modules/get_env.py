import gym


def get_env(env_id):
    """Return train and val env"""
    if env_id == "CartPole-v0":
        env = gym.make(env_id)
        val_env = gym.make(env_id)
    elif env_id == "Alien-v0":
        env = gym.make(env_id)
        val_env = gym.make(env_id)
    else:
        raise ValueError("Unsupported Env Name")

    return env, val_env
