import hydra
from train_heinrich.task_registry import task_registry

@hydra.main(config_path="configs", config_name="train_config")
def train(cfg):
    env, env_cfg = task_registry.make_env(name=cfg.task, args=cfg)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=cfg.task, args=cfg)
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

if __name__ == '__main__':
    train()