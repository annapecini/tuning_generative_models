import sys
import os
from pathlib import Path
import pandas as pd
sys.path.append(os.path.join(Path(__file__).parents[1], ''))
import lib
from sdv.tabular import TVAE
from sdv.evaluation import evaluate
from rdt.transformers import FloatFormatter, OneHotEncoder, UnixTimestampEncoder

ds_name = "totalturnover"
prefix = "tvae"
parent_path = Path(f'exp/{ds_name}/tvae_best/')
real_data_path = os.path.join(Path(__file__).parents[1], 'totalturnover.csv')

data = pd.read_csv(real_data_path, parse_dates=['date'])
# Process real data
data['total_turnover'] = data['total_turnover'].apply(lambda x: x / 100)

# Encode holidays in 0/1
data['holiday_school'] = data['holiday_school'].fillna(0)
data.loc[data['holiday_school'] != 0, 'holiday_school'] = 1

data['holiday_public'] = data['holiday_public'].fillna(0)
data.loc[data['holiday_public'] != 0, 'holiday_public'] = 1


def train_tvae():
    parameters = lib.load_json(os.path.join(parent_path, 'parameters.json'))

    field_transformers = {"date": UnixTimestampEncoder(),
            "total_turnover": FloatFormatter(),
            "whi_temp": FloatFormatter(),
            "whi_temp_min": FloatFormatter(),
            "whi_temp_max": FloatFormatter(),
            "whi_feels_like": FloatFormatter(),
            "whi_pressure": FloatFormatter(),
            "whi_humidity": FloatFormatter,
            "whi_clouds": FloatFormatter(),
            "whi_rain": FloatFormatter(),
            "holiday_school": OneHotEncoder(),
            "holiday_public": OneHotEncoder()
    }

    model = TVAE(field_transformers=field_transformers, **parameters)
    model.fit(data)
    model.save(os.path.join(parent_path, 'model.pkl'))


def sample_tvae():
    loaded = TVAE.load(os.path.join(parent_path, 'model.pkl'))
    new_data = loaded.sample(num_rows=data.shape[0])

    # Sort by date
    new_data.sort_values(by=['date'], inplace=True)
    
    # Save dataframe as csv
    new_data.to_csv(os.path.join(Path(__file__).parents[1], 'synthetic_tvae_totalturnover.csv'), index=True)


def main():
    # train_tvae()
    sample_tvae()


if __name__ == '__main__':
    main()