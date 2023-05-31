import sys
import os
from pathlib import Path
import pandas as pd
sys.path.append(os.path.join(Path(__file__).parents[1], ''))
from sdv.tabular import TVAE
from sdv.evaluation import evaluate
from rdt.transformers import FloatFormatter, OneHotEncoder, UnixTimestampEncoder
import optuna
import lib


ds_name = "nike"
prefix = "tvae"
parent_path = Path("exp/nike/")

real_data_path = os.path.join(Path(__file__).parents[1], 'nike_sales_short.csv')

# os.makedirs(exps_path, exist_ok=True)

def objective(trial):
    data = pd.read_csv(real_data_path, parse_dates=['date'])
    data = data.dropna(subset=['value'])
    # Process real data
    # data['total_turnover'] = data['total_turnover'].apply(lambda x: x / 100)

    # Encode holidays in 0/1
    # data['holiday_school'] = data['holiday_school'].fillna(0)
    # data.loc[data['holiday_school'] != 0, 'holiday_school'] = 1

    # data['holiday_public'] = data['holiday_public'].fillna(0)
    # data.loc[data['holiday_public'] != 0, 'holiday_public'] = 1

    # Hyperparameters
    embedding_dim = trial.suggest_categorical("embedding_dim", [128, 256, 512])
    nr_layers = trial.suggest_categorical("nr_layers", [2,3])
    dims = trial.suggest_categorical("dims", [128,256])
    l2scale = trial.suggest_float("l2scale", 0.00001, 0.00005, step=0.00001)
    batch_size = trial.suggest_categorical("batch_size", [100, 200, 500])
    epochs = trial.suggest_categorical("epochs", [300, 700, 1000, 2000])

    # field_transformers = {"date": UnixTimestampEncoder(),
    #    "cal_holiday": OneHotEncoder(),
    #    "weather_temp": FloatFormatter(),
    #    "weather_rain_1h": FloatFormatter(),
    #    "weather_snow_1h": FloatFormatter(),
    #    "weather_clouds_all": FloatFormatter(),
    #    "weather_main": OneHotEncoder(),
    #    "weather_description": OneHotEncoder(),
    #    "traffic_volume": FloatFormatter()
    #}
    field_transformers = {
            "date": UnixTimestampEncoder(),
            "value": FloatFormatter(),
            "GPD_per_capita": FloatFormatter(),
            "eu_prod_index": FloatFormatter()
    }

    trial.set_user_attr("embedding_dim", embedding_dim)
    trial.set_user_attr("l2scale", l2scale)
    trial.set_user_attr("batch_size", batch_size)
    trial.set_user_attr("epochs", epochs)
    trial.set_user_attr("compress_dims", [dims] * nr_layers)
    trial.set_user_attr("decompress_dims", [dims] * nr_layers)

    # Fit data
    model = TVAE(embedding_dim=trial.user_attrs["embedding_dim"],
                 compress_dims=trial.user_attrs["compress_dims"],
                 decompress_dims=trial.user_attrs["decompress_dims"],
                 l2scale=trial.user_attrs["l2scale"],
                 batch_size=trial.user_attrs["batch_size"],
                 epochs=trial.user_attrs["epochs"],
                 field_transformers=field_transformers
                 )
    model.fit(data)

    new_data = model.sample(num_rows=len(data))
    print(new_data.head())
    return evaluate(new_data, data)


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100, show_progress_bar=True)
best_trial = study.best_trial

os.makedirs(parent_path / 'tvae_best', exist_ok=True)
lib.dump_json(optuna.importance.get_param_importances(study), parent_path / 'tvae_best/importance.json')
lib.dump_json(best_trial.user_attrs, parent_path / 'tvae_best/parameters.json')

# print(best_trial)
print(best_trial.user_attrs)
