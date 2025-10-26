import optuna

boxing_db = "sqlite:///boxing_optuna/boxing_optuna.db"
study_name = "boxing"

study = optuna.load_study(study_name, storage=boxing_db)

# We just pick the best 3
top_3_trials = sorted(study.trials, key=lambda t: t.value, reverse=True)[:3]

for i, trial in enumerate(top_3_trials):
    print(f"Trail number: {trial.number}")
    print(f"Value: {trial.value}")
    print("Hyperparameters obtained:")
    for key, value in trial.params.items():
        print(f"{key}: {value}")
    print("\n")