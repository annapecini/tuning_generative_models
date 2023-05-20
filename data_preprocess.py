import pandas as pd
import numpy as np
import os
import datetime
from sklearn.model_selection import train_test_split

dataset_name = "metro"

def date_to_days(date):
    days = (date - datetime.datetime(1970, 1, 1)).days
    return days


def main():
    df = pd.read_csv('Metro_Interstate_Traffic_Volume_short.csv')

    df = df.dropna(subset=['traffic_volume'])

    # Convert date to datetime type
    df["date"] = pd.to_datetime(df["date"], format='%Y-%m-%d %H:%M:%S')
    df["hour"] = pd.to_datetime(df["date"]).dt.hour


    # Convert date to number of days
    df['date'] = df['date'].apply(lambda x: date_to_days(x))


    # Convert all numeric types from float64 to float32
    df['weather_temp'] = pd.to_numeric(df['weather_temp'], downcast="float")
    df['weather_rain_1h'] = pd.to_numeric(df['weather_rain_1h'], downcast="float")
    df['weather_snow_1h'] = pd.to_numeric(df['weather_snow_1h'], downcast="float")
    df['weather_clouds_all'] = pd.to_numeric(df['weather_clouds_all'], downcast="float")
    df['traffic_volume'] = pd.to_numeric(df['traffic_volume'], downcast="float")


    # Get train, test and val sets
    X = df.drop(['traffic_volume'], axis=1).values.tolist()
    y = df['traffic_volume'].tolist()


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=1)


    # Save npy files
    X_num_train = np.array([np.hstack((a_1d[:1], a_1d[2:6])) for a_1d in X_train])
    X_cat_train = np.array([np.hstack((a_1d[1:2], a_1d[6:9])) for a_1d in X_train])

    X_num_test = np.array([np.hstack((a_1d[:1], a_1d[2:6])) for a_1d in X_test])
    X_cat_test = np.array([np.hstack((a_1d[1:2], a_1d[6:9])) for a_1d in X_test])

    X_num_val = np.array([np.hstack((a_1d[:1], a_1d[2:6])) for a_1d in X_val])
    X_cat_val = np.array([np.hstack((a_1d[1:2], a_1d[6:9])) for a_1d in X_val])

    np.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', dataset_name, 'X_num_train.npy'),
            X_num_train)
    np.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', dataset_name, 'X_cat_train.npy'),
            X_cat_train)
    np.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', dataset_name, 'y_train.npy'), y_train)

    np.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', dataset_name, 'X_num_test.npy'),
            X_num_test)
    np.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', dataset_name, 'X_cat_test.npy'),
            X_cat_test)
    np.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', dataset_name, 'y_test.npy'), y_test)

    np.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', dataset_name, 'X_num_val.npy'),
            X_num_val)
    np.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', dataset_name, 'X_cat_val.npy'),
            X_cat_val)
    np.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', dataset_name, 'y_val.npy'), y_val)
    # np.save('X_num_train.npy', X_num_train)
    # np.save('X_cat_train.npy', X_cat_train)
    # np.save('y_train.npy', y_train)
    #
    # np.save('X_num_test.npy', X_num_test)
    # np.save('X_cat_test.npy', X_cat_test)
    # np.save('y_test.npy', y_test)
    #
    # np.save( 'X_num_val.npy', X_num_val)
    # np.save('X_cat_val.npy', X_cat_val)
    # np.save('y_val.npy', y_val)


if __name__ == '__main__':
    main()
    # X_num_train = np.load('X_num_test.npy', allow_pickle=True)
    # X_cat_train = np.load('X_cat_test.npy', allow_pickle=True)
    # y_train = np.load('y_train.npy')
    # print(X_num_train.shape)
    # print(X_cat_train.shape)

