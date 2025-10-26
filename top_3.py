import os
import json
import random
import gymnasium as gym
import ale_py  # asegura el registro de ALE en gym
import numpy as np
from contextlib import contextmanager
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box
from torch.utils.tensorboard import SummaryWriter

from deustorl.common import evaluate_policy, EpsilonGreedyPolicy, max_policy
from deustorl.sarsa import Sarsa
from deustorl.qlearning import QLearning
from deustorl.expected_sarsa import ExpectedSarsa
from deustorl.helpers import DiscretizedObservationWrapper

# ============================
# Config
# ============================
TXT_PATH = "boxing_top_3_trials.txt"
ENV_NAME = "ALE/Boxing-v5"
SEED_BASE = 88
N_STEPS_TOTAL = 100_000         # pon 1_000_000 si quieres entrenar más
ALGORITHMS = ["Sarsa", "Qlearning", "ExpectedSarsa"]

# Un único “root” para logs nuevos
LOG_ROOT = "./tensorboard_logs_top_3"   # <-- Arranca TB apuntando a LOG_ROOT + "/logs"
LOG_SUBFOLDER = "logs"                  # quedará: tensorboard_logs_top_3/logs/<run_name>/

# ============================
# Wrappers
# ============================
class RamFeatureWrapper(ObservationWrapper):
    def __init__(self, env, num_features=8):
        super().__init__(env)
        self.num_features = num_features
        self.observation_space = Box(low=0, high=255, shape=(num_features,), dtype=np.uint8)
    def observation(self, obs):
        return obs[:self.num_features]

# ============================
# Utils
# ============================
def load_top_trials_from_txt(path=TXT_PATH):
    """Parsea el TXT (formato tal cual lo guardaste) → lista de dicts con hiperparámetros."""
    with open(path, "r") as f:
        lines = f.readlines()

    trials = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("Trial number:"):
            tnum = int(line.split(": ")[1])
            val = float(lines[i + 1].split(": ")[1])  # Value
            i += 3  # salta "Hyperparameters obtained:"
            entry = {"trial_number": tnum, "value": val}
            while i < len(lines) and lines[i].strip():
                k, v = lines[i].strip().split(": ")
                try:
                    entry[k] = float(v) if ("." in v or "e" in v.lower()) else int(v)
                except ValueError:
                    entry[k] = v
                i += 1
            trials.append(entry)
        i += 1
    return trials

def expand_with_algorithms_if_missing(trials):
    """Si el TXT no incluye 'algorithm', crea 3 runs por combinación (Sarsa/Qlearning/ExpectedSarsa)."""
    expanded = []
    for t in trials:
        algo = t.get("algorithm", None)
        if algo in ALGORITHMS:
            expanded.append(t)
        else:
            for a in ALGORITHMS:
                tt = dict(t)
                tt["algorithm"] = a
                expanded.append(tt)
    return expanded

def make_algo(name, env):
    if name == "Sarsa":
        return Sarsa(env)
    if name == "Qlearning":
        return QLearning(env)
    if name == "ExpectedSarsa":
        return ExpectedSarsa(env)
    raise ValueError(f"Algoritmo desconocido: {name}")

@contextmanager
def pushd(new_dir: str):
    """Cambia temporalmente el cwd. Útil para que cualquier 'logs/' relativo caiga dentro de LOG_ROOT."""
    prev = os.getcwd()
    os.makedirs(new_dir, exist_ok=True)
    os.chdir(new_dir)
    try:
        yield
    finally:
        os.chdir(prev)

# ============================
# “Trial” runner (similar a hyperparameter_analysis, pero sin Optuna)
# ============================
def run_with_params(run_idx, params):
    """
    Entrena y evalúa una combinación concreta.
    Estructura de logs: tensorboard_logs_top_3/logs/<Algorithm>_<TrialNumber>/
    """
    run_name = f"{params['algorithm']}_{params['trial_number']}"
    # Todo lo que escriba internamente algo como "logs/..." quedará bajo LOG_ROOT/
    with pushd(LOG_ROOT):
        # Crear una única carpeta de logs para todos los trainings y una subcarpeta por run
        main_logs_dir = LOG_SUBFOLDER
        os.makedirs(main_logs_dir, exist_ok=True)
        log_dir = os.path.join(main_logs_dir, run_name)
        os.makedirs(log_dir, exist_ok=True)

        writer = SummaryWriter(log_dir=log_dir)

        # ====== Entorno ======
        env = gym.make(
            ENV_NAME,
            obs_type="ram",
            render_mode="rgb_array",  # evita error si tu learn() llama a env.render()
            frameskip=4
        )
        env = RamFeatureWrapper(env, num_features=8)
        env = DiscretizedObservationWrapper(env, n_bins=4)

        # ====== Hiperparámetros ======
        algorithm_name    = params["algorithm"]
        learning_rate     = float(params["learning_rate"])
        discount_factor   = float(params["discount_factor"])
        epsilon           = float(params["epsilon"])
        lr_decay          = float(params["lr_decay"])
        lr_episodes_decay = int(params["lr_episodes_decay"])

        # ====== Algoritmo y política ======
        algo = make_algo(algorithm_name, env)
        policy = EpsilonGreedyPolicy(epsilon=epsilon)

        # ====== Semillas ======
        seed = SEED_BASE + run_idx
        env.reset(seed=seed)
        env.action_space.seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        # ====== Entrenamiento ======
        algo.learn(
            policy,
            n_steps=N_STEPS_TOTAL,
            discount_rate=discount_factor,
            lr=learning_rate,
            lrdecay=lr_decay,
            n_episodes_decay=lr_episodes_decay,
            verbose=False
        )

        # ====== Evaluación ======
        avg_reward, avg_steps = evaluate_policy(env, algo.q_table, max_policy, n_episodes=20)

        # ====== Logging (un único writer por run) ======
        writer.add_scalar("eval/avg_reward", avg_reward, run_idx)
        writer.add_scalar("eval/avg_steps",  avg_steps,  run_idx)
        writer.add_hparams({
            "algorithm": algorithm_name,
            "trial_number": params["trial_number"],
            "learning_rate": learning_rate,
            "discount_factor": discount_factor,
            "epsilon": epsilon,
            "lr_decay": lr_decay,
            "lr_episodes_decay": lr_episodes_decay,
            "value_from_txt": params.get("value", None),
        }, {"avg_reward": avg_reward})
        writer.flush()
        writer.close()
        env.close()

        print(f"[RESULT] {run_name}: avg_reward={avg_reward:.3f} | avg_steps={avg_steps:.1f}")
        return {
            "run_name": run_name,
            "trial_number": params["trial_number"],
            "algorithm": algorithm_name,
            "avg_reward": float(avg_reward),
            "avg_steps": float(avg_steps),
            "learning_rate": learning_rate,
            "discount_factor": discount_factor,
            "epsilon": epsilon,
            "lr_decay": lr_decay,
            "lr_episodes_decay": lr_episodes_decay,
        }

# ============================
# Main
# ============================
if __name__ == "__main__":
    # Limpia solo nuestro root (no toca tus antiguos ./logs)
    os.system(f"rm -rf {LOG_ROOT}")
    os.makedirs(LOG_ROOT, exist_ok=True)

    # Cargar top-3 del TXT y expandir si falta 'algorithm'
    base_trials = load_top_trials_from_txt(TXT_PATH)
    runs = expand_with_algorithms_if_missing(base_trials)

    # Guarda el plan de ejecución
    with open(os.path.join(LOG_ROOT, "top_trials_enriched.json"), "w") as f:
        json.dump(runs, f, indent=2)

    # Ejecutar todos los runs (3 o 9 según el TXT)
    results = []
    for idx, params in enumerate(runs, start=1):
        results.append(run_with_params(idx, params))

    # Resumen
    with open(os.path.join(LOG_ROOT, "summary_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\n✅ Terminado.")