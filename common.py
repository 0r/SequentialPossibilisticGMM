import numpy as np
import pandas as pd
import pickle

from sklearn.decomposition import PCA

def read_sensor_data(fname, parse_dates=None):
    path = './data/sensor/'
    if parse_dates is None:
        df = pd.read_csv(path + fname)
    else:
        df = pd.read_csv(path + fname, parse_dates=parse_dates)
    return df

def pca_3_components(df):
    pca = PCA(n_components=3)
    df_reduced = pca.fit_transform(df)
    df_reduced = pd.DataFrame(df_reduced,columns=['PC1','PC2','PC3'])
    return df_reduced

def init_stream(df,ndays=30):
    init_data   = df[0:ndays].to_numpy()
    stream_data = df[ndays:].to_numpy()
    return init_data, stream_data

def write_model_pickle(model,fname):
    with open('./model/'+fname, 'wb') as file:
        pickle.dump(model, file)

def make_date_column(df, cname):
    df[cname] = pd.to_datetime(df[cname])
    ddf = df.copy()
    ddf.loc[:, 'date'] = ddf[cname].dt.date
    return ddf

def get_daily_mean(df):
    df = df.groupby('date', as_index = False).mean(numeric_only=True)
    return df

def fill_missing_dates(df):
    sd = df['date'].min()
    ed = df['date'].max()
    idx = pd.date_range(sd,ed)
    df.index = pd.DatetimeIndex(df['date'])
    df.set_index('date', inplace = True)
    df = df.reindex(idx, fill_value=np.nan)
    df = df.reset_index()
    df.rename(columns={'index':'date'}, inplace=True)
    return df