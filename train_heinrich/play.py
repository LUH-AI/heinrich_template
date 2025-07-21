import os
import hydra
from functools import partial
from mighty.mighty_runners.factory import get_runner_class

from train_heinrich.make_isaac_gym_env import make_env

LEGGED_GYM_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


@hydra.main(config_path="configs", config_name="play_config")
def play(cfg):
    # Make env
    cfg.environment.test = True
    env = make_env(cfg.environment)
    eval_env_constructor = partial(make_env, cfg.environment)
    eval_default = cfg.n_episodes_eval

    # Load policy
    runner_cls = get_runner_class(cfg.runner)
    runner = runner_cls(cfg, env, eval_env_constructor, eval_default)
    runner.agent.load(cfg.checkpoint_path)

    # Run eval
    eval_metrics = runner.evaluate()
    print(eval_metrics)


if __name__ == "__main__":
    play()
