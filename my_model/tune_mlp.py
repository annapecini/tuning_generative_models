import sys
import argparse
import os
from pathlib import Path
from sdv.evaluation import evaluate
import optuna
sys.path.append(os.path.join(Path(__file__).parents[1], ''))
import lib
import pandas as pd
from diffusion_model import *


ds_name = "totalturnover"
parent_path = Path(f'exp/{ds_name}/')
prefix = "diffusion_mlp"

real_data_path = os.path.join(Path(__file__).parents[1], 'totalturnover.csv')
synthetic_data_path = 'synthetic_my_model.csv'

def objective(trial):

    data = pd.read_csv(real_data_path, parse_dates=['date'])
    # Process real data
    # Drop categorical values
    data.drop(columns=['total_turnover', 'holiday_school', 'holiday_public'], inplace=True)

    # Hyperparameters
    dropouts = trial.suggest_discrete_uniform("dropouts", 0.0, 0.5, 0.1)
    lr = trial.suggest_loguniform('lr', 0.00001, 0.003)
    batch_size = trial.suggest_categorical("batch_size", [5, 10, 15, 20, 50])
    epochs = trial.suggest_categorical("epochs", [100, 300, 500, 1000])
    nr_layers = trial.suggest_categorical("nr_layers", [2, 3])
    dims = trial.suggest_categorical("dims", [128, 256])

    trial.set_user_attr("dropouts", dropouts)
    trial.set_user_attr("lr", lr)
    trial.set_user_attr("batch_size", batch_size)
    trial.set_user_attr("epochs", epochs)
    trial.set_user_attr("d_layers", [dims] * nr_layers)

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.epochs = trial.user_attrs["epochs"]
    args.batch_size = trial.user_attrs["batch_size"]
    args.lr = trial.user_attrs["lr"]
    args.d_layers = trial.user_attrs["d_layers"]
    args.dropouts = trial.user_attrs["dropouts"]
    args.device = "cuda"
    args.dataset_path = os.path.join(Path(__file__).parents[1], 'data', 'totalturnover/')
    # args.dataset_path = r"/Users/apecini/Documents/thesis/master-thesis/data/totalturnover/"
    args.num_features = 9

    print(args)
    # Fit data
    train(args)

    new_data = pd.read_csv(synthetic_data_path, parse_dates=['date'])
    return evaluate(new_data, real_data=data)


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20, show_progress_bar=True)
best_trial = study.best_trial

os.makedirs(parent_path / f'{prefix}_best', exist_ok=True)
lib.dump_json(optuna.importance.get_param_importances(study), parent_path / f'{prefix}_best/importance.json')
lib.dump_json(best_trial.user_attrs, parent_path / f'{prefix}_best/parameters.json')

# print(best_trial)
print(best_trial.user_attrs)