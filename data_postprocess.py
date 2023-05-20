import pandas as pd
import os
from pathlib import Path
import numpy as np

def main():
    # Get path of X_num, X_cat and y
    dir = os.path.join(Path(__file__).parents[0], 'exp', 'metro', 'ddpm_tune_best/')
    X_num_train = np.load(os.path.join(dir, 'X_num_unnorm.npy'), allow_pickle = True)
    X_cat_train = np.load(os.path.join(dir, 'X_cat_train.npy'), allow_pickle=True)
    y_train = np.load(os.path.join(dir, 'y_train.npy'))

    # Read X_num
    df_num = pd.DataFrame(X_num_train, columns=['date',
                                                'weather_temp',
                                                'weather_rain_1h',
                                                'weather_snow_1h',
                                                'weather_clouds_all',
                                                'traffic_volume'
                                                ])

    print(df_num.nunique())

    # # Convert number of days to date
    # df_num["date"] = pd.to_datetime(df_num["date"], unit="d")
    #
    # # Read X_cat
    # df_cat = pd.DataFrame(X_cat_train, columns=['cat_holiday',
    #                                             'weather_main',
    #                                             'weather_description',
    #                                             'hour'])
    #
    # # Read y
    # df_y = pd.DataFrame(y_train, columns=['traffic_volume'])
    #
    # # Concat X_num, X_cat and y
    # df = pd.concat([df_num, df_cat, df_y], axis=1)
    #
    # # Create datetime column
    # df['date'] = df['date'].dt.strftime('%Y-%m-%d') + ' ' + df['hour'].astype(str) + ':00:00'
    #
    #
    # # Convert datetime column to datetime format
    # df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:%S')
    # # df.drop_duplicates(subset=["date"], inplace=True)
    # df.drop('hour', axis=1, inplace=True)
    # # Sort by date
    # df.sort_values(by=['date'], inplace=True)
    #
    # column_order = ['date',
    #                 'cat_holiday',
    #                 'weather_temp',
    #                 'weather_rain_1h',
    #                 'weather_snow_1h',
    #                 'weather_main',
    #                 'weather_description',
    #                 'traffic_volume'
    #                 ]
    #
    # df = df[column_order]
    #
    # # Save dataframe as csv
    # df.to_csv(os.path.join(Path(__file__).parents[0], 'exp', 'metro', 'ddpm_tune_best', 'synthetic_ddpm_metro.csv'), index=False)


if __name__ == '__main__':
    main()