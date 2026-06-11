import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
import torch
import marl
import logging
import dotenv

from marl import ReplayMemory
from marl.algos.dqn import DQN, SoftUpdate
from marlenv import MARLEnv, MultiDiscreteSpace
from typing import Any, Literal, Optional, cast
from lle import LLE
import os
from create_experiments import make_mixer

N_STEPS = 1_000_000
 
def suggest_recurrent_dqn(trial: optuna.Trial,
    env: MARLEnv[MultiDiscreteSpace],
    mixing: Optional[Literal["vdn", "qmix"]] = "vdn",
    gamma: float = 0.95,
    noisy: bool = False,
    use_vbe: bool = False,
    memory: Optional[ReplayMemory[Any]] = None,):
    mixer = make_mixer(env, mixing)
    if len(env.observation_shape) == 1:
        # marl.training.DQN(mixer=marl.nn.mixers.VDN.from_env(env), **dqn_params)
        qnetwork = marl.nn.model_bank.qnetworks.QCRNN.from_env(env)
    elif len(env.observation_shape) == 3:
        # marl.training.DQN(mixer=marl.nn.mixers.VDN.from_env(env), **dqn_params)
        qnetwork = marl.nn.model_bank.qnetworks.QCRNN.from_env(env)
    else:
        raise NotImplementedError(f"Observation shape {env.observation_shape} not supported")
    ir = None
    if noisy:
        policy = marl.policy.ArgMax()
    else:
        policy = marl.policy.EpsilonGreedy.linear(1.0, 0.05, n_steps=200_000)
    return DQN(
        qnetwork=qnetwork,
        train_policy=policy,
        memory_size=5000,
        optimiser_type="adam",
        double_qlearning=True,
        target_updater=SoftUpdate(0.01),
        lr=5e-4,
        batch_size=16,
        train_interval=(1, "episode"),
        gamma=gamma,
        mixer=mixer,
        grad_norm_clipping=10,
        ir_module=ir,
        vbe=None,
    )  


def objective(trial: optuna.Trial, algo: Literal["vdn", "qmix"], obs: Literal["layered", "partial3x3", "partial5x5", "partial7x7"] = "layered"):
    
    env = marl.env.LLEConfig(6, obs_type=obs, state_type="state")
    trainer = suggest_recurrent_dqn(trial, env, algo)

    exp = marl.Experiment(
        env,
        n_steps=N_STEPS,
        trainer=trainer,
        logdir=os.path.join("logs", f"optuna-{algo}-{trial.number}"),
    )
    exp.run(4, "scatter", quiet=True, n_parallel=4, n_tests=30, device=trial.number % torch.cuda.device_count())
    result = exp.get_experiment_results()
    df = result["Test"]
    score = df.select("mean-exit_rate").last().collect().item()
    return score


def suggest_dqn(trial: optuna.Trial, env: MARLEnv[MultiDiscreteSpace]) -> dict[str, Any]:
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_int("batch_size", 16, 256, step=16)
    gamma = trial.suggest_float("gamma", 0.9, 1.0, step=0.01)
    epsilon_start = trial.suggest_float("epsilon_start", 0.9, 1.0)
    epsilon_end = trial.suggest_float("epsilon_end", 0.001, 0.1, log=True)
    epsilon_decay = trial.suggest_int("epsilon_decay", 1000, 500_000, step=1000)
    grad_norm_clipping = trial.suggest_float("grad_norm_clipping", 0.5, 50.0, step=0.5)
    memory_size = trial.suggest_int("memory_size", 5_000, 100_000, step=5_000)
    double_qlearning = trial.suggest_categorical("double_qlearning", [True, False])
    update_type = trial.suggest_categorical("update_type", ["soft", "hard"])

    return dict(
        qnetwork= marl.nn.model_bank.qnetworks.QCNN.from_env(env, mlp_sizes=(mlp_size_1, mlp_size_2)),
        train_policy=marl.policy.EpsilonGreedy.linear(epsilon_start, epsilon_end, epsilon_decay),
        memory=marl.models.TransitionMemory(memory_size),
        optimiser_type=cast(Literal["adam", "rmsprop"], optimizer_type),
        gamma=gamma,
        batch_size=batch_size,
        lr=learning_rate,
        grad_norm_clipping=grad_norm_clipping,
        target_updater=target_updater,
        double_qlearning=double_qlearning,
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
    for algo in ('vdn', 'qmix'):
        try:
            study = optuna.create_study(
                direction="maximize",
                study_name=f"{algo.upper()} - {'no shaping'}",
                storage=JournalStorage(JournalFileBackend("optuna_study.journal")),
                load_if_exists=True,
            )
            n_trials = 5
            study.optimize(lambda trial: objective(trial, algo=algo), n_trials=n_trials, n_jobs=8)
        except KeyboardInterrupt:
            pass
        except Exception as e:
            logging.error("An error occurred during optimization.", exc_info=e)
