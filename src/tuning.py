from create_experiments import make_mixer
import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
import torch

import marl
from marl.algos.dqn import DQN
from marl.algos.qtarget_updater import SoftUpdate, HardUpdate

import logging
import dotenv

from typing import Any, Literal, Optional

from marlenv import MARLEnv, MultiDiscreteSpace
from marl.nn.model_bank import qnetworks
from typing import cast, Literal, Any
import lle
from lle import LLE, env
import os

N_STEPS = 1_000_000
SHAPING = False

def make_recurrent_dqn(
    env,
    mixing,
    train_policy,
    memory_size,
    train_interval,
    gamma,
    batch_size,
    lr,
    grad_norm_clipping,
    target_updater,
    test_policy):
    
    mixer = make_mixer(env, mixing)
    if len(env.observation_shape) == 1:
        qnetwork = marl.nn.model_bank.qnetworks.QCRNN.from_env(env)
    elif len(env.observation_shape) == 3:
        qnetwork = marl.nn.model_bank.qnetworks.QCRNN.from_env(env)
    else:
        raise NotImplementedError(f"Observation shape {env.observation_shape} not supported")
    ir = None
    return DQN(
        qnetwork=qnetwork,
        train_policy=train_policy,
        memory_size=memory_size,
        optimiser_type="adam",
        double_qlearning=True,
        target_updater=target_updater,
        lr=lr,
        batch_size=batch_size,
        train_interval=train_interval,
        gamma=gamma,
        mixer=mixer,
        grad_norm_clipping=grad_norm_clipping,
        ir_module=ir,
        vbe=None,
    )  


def objective(trial: optuna.Trial, algo: Literal["vdn", "qmix"]):
    env = lle.level(6).obs_type("layered").state_type("state")
    if SHAPING:
        env = env.pbrs(gamma=1.0, reward_value=1.0, lasers_to_reward=[(4, 0), (6, 12)])
    env = env.builder().agent_id().time_limit(78).build()
    match algo:
        case "vdn":
            dqn_params = suggest_dqn(trial, env)
            trainer = make_recurrent_dqn(env=env, mixing="vdn", **dqn_params)
        case "qmix":
            dqn_params = suggest_dqn(trial, env)
            trainer = make_recurrent_dqn(env=env, mixing="qmix", **dqn_params)
        case other:
            raise NotImplementedError(f"Algorithm {other} not implemented yet.")
    exp = marl.Experiment(
        env=env,
        trainer=trainer,
        n_steps=N_STEPS,
        logdir=os.path.join("logs", f"optuna-{algo}-{trial.number}"),
        loggers = ["csv"]
    )
    exp.run(4, "scatter", quiet=False, n_jobs=4, n_tests=30, device=trial.number % torch.cuda.device_count())
    result = exp.get_experiment_results()
    df = result["Test"]
    score = df.select("mean-exit_rate").last().collect().item()
    return score


def suggest_dqn(trial: optuna.Trial, env: MARLEnv[MultiDiscreteSpace]) -> dict[str, Any]:
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_int("batch_size", 4, 32, step=4)
    train_interval = trial.suggest_int("train_interval", 1, 10, step=1)
    gamma = trial.suggest_float("gamma", 0.9, 1.0, step=0.01)
    epsilon_start = trial.suggest_float("epsilon_start", 0.9, 1.0)
    epsilon_end = trial.suggest_float("epsilon_end", 0.001, 0.1, log=True)
    epsilon_decay = trial.suggest_int("epsilon_decay", 1000, 500_000, step=1000)
    grad_norm_clipping = trial.suggest_float("grad_norm_clipping", 0.5, 50.0, step=0.5)
    memory_size = trial.suggest_int("memory_size", 1_000, 10_000, step=1_000)
    update_type = trial.suggest_categorical("update_type", ["soft", "hard"])
    if update_type == "soft":
        tau = trial.suggest_float("tau", 1e-3, 1e-1, log=True)
        target_updater = SoftUpdate(tau)
    else:
        target_update_interval = trial.suggest_int("target_update_interval", 50, 1_000, step=50)
        target_updater = HardUpdate(target_update_interval)

    return dict(
        train_policy=marl.policy.EpsilonGreedy.linear(epsilon_start, epsilon_end, epsilon_decay),
        memory_size=memory_size,
        train_interval = (train_interval, "episode"),
        gamma=gamma,
        batch_size=batch_size,
        lr=learning_rate,
        grad_norm_clipping=grad_norm_clipping,
        target_updater=target_updater,
        test_policy=marl.policy.ArgMax(),
    )


if __name__ == "__main__":
    dotenv.load_dotenv()
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        handlers=[logging.FileHandler("tuning.log", mode="a"), logging.StreamHandler()],
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    for algo in ("vdn", "qmix"):
        try:
            study = optuna.create_study(
                direction="maximize",
                study_name=f"{algo} - {'shaping' if SHAPING else 'no shaping'}",
                storage=JournalStorage(JournalFileBackend("optuna_study.journal")),
                load_if_exists=True,
            )
            n_trials = 20
            study.optimize(lambda trial: objective(trial, algo=algo), n_trials=n_trials, n_jobs=5)
        except KeyboardInterrupt:
            pass
        except Exception as e:
            logging.error("An error occurred during optimization.", exc_info=e)
