import numpy as np
import urllib.request, os, pandas as pd

def windowed_data(x_norm, lead_time=5, window_size=6):
    '''Prepare windowed data with a given window size
    - columns:`[Y(T+lead_time), Y(T+0), ...,Y(T-window_size+1)]`

    Arg-s:
    - x_norm : input data
    - lead_time : lead time, Y(T+lead_time) that you are trying to predict.
    - window_size : number of past inputs, [ Y(T+0), ...,Y(T-window_size+1)],
    where Y(t) =  x_norm[t]
    '''
    start_time = window_size-1
    X = [x_norm[start_time+lead_time:].reshape(-1,1) ] # Y(T+lead_time)
    # append Y(T+0) to Y(T-window_size+1)
    for w in range(window_size,0,-1):
        X.append(x_norm[w-1:x_norm.shape[0] - lead_time - window_size+w].reshape(-1,1))
    # concatenate Y(T+lead_time), Y(T+0),...,Y(T-window_size+1)
    return np.concatenate(X,axis=1)

def windowed_diff_data(x_norm, lead_time=5, window_size=6):
    '''Prepare windowed data with a given window size
    - 1st and 2nd columns:`[Y(T+lead_time), Y(T+0),]
    - 3rd and subsequent columns are differences:
    `[, Y(T+0)-Y(T-1), Y(T-1)-Y(T-2),..., Y(T-window_size+2)-Y(T-window_size+1)]`
    '''
    X = windowed_data(x_norm, lead_time=lead_time, window_size=window_size)
    X[:,2:] = X[:,1:-1]-X[:,2:]
    # concatenate Y(T+lead_time), Y(T+0), differences
    return X


# wind farm locations
locations = ['guitrancourt', 'lieusaint', 'lvs-pussay', 'parc-du-gatinais',
 'arville', 'boissy-la-riviere', 'angerville-1', 'angerville-2']
# angerville-2 is "Les Pointes" wind farm

def download_forecasts_latest():
    '''Download latest wind forecasts. Saves all files to
    `./datasets/model1/` and `./datasets/model2/` for two forecast models.'''
    # Download latest wind forecasts
    base_url_ = 'https://ai4impact.org/P003/'
    data_path = os.path.relpath('datasets/')
    for l in locations:
        # model 1 forecast
        file_url_ = base_url_+l+'.csv'
        file_name,_ = urllib.request.urlretrieve(
            file_url_,filename=os.path.join(data_path,'model1',l+'.csv'))

        # model 2 forecast
        file_url_ = base_url_+l+'-b.csv'
        file_name,_ = urllib.request.urlretrieve(
            file_url_,filename=os.path.join(data_path,'model2',l+'-b.csv'))
    print(f'Downloaded from:\n`{base_url_}``\nsaved to:\n{data_path}')

def download_forecasts_all():
    '''
    Download all (historical and latest) wind forecasts.

    Saves all files to `./datasets/model1/`, `./datasets/model2/`,
    `./datasets/historical1/`, and `./datasets/historical2/`
    for two forecast models.
    '''

    data_path = os.path.relpath('datasets/')

    # Latest
    print('Latest wind forecasts:')
    base_url_ = 'https://ai4impact.org/P003/'
    for l in locations:
        # model 1 forecast
        file_url_ = base_url_+l+'.csv'
        file_name,_ = urllib.request.urlretrieve(
            file_url_,filename=os.path.join(data_path,'model1',l+'.csv'))
        # model 2 forecast
        file_url_ = base_url_+l+'-b.csv'
        file_name,_ = urllib.request.urlretrieve(
            file_url_,filename=os.path.join(data_path,'model2',l+'-b.csv'))
    print(f'Downloaded latest forecasts from\n`{base_url_}`\nsaved to\n{data_path}\n')

    # Historical (past) forecasts
    print('Historical wind forecasts:')
    base_url_2 = 'https://ai4impact.org/P003/historical/'
    for l in locations:
        file_url_ = base_url_2+l+'.csv'
        # model 1 historical forecasts
        file_name,_ = urllib.request.urlretrieve(
            file_url_,filename=os.path.join(data_path,'historical1',l+'.csv'))
        # model 2 historical forecasts
        file_url_ = base_url_2+l+'-b.csv'
        file_name,_ = urllib.request.urlretrieve(
            file_url_,filename=os.path.join(data_path,'historical2',l+'-b.csv'))
    print(f'Downloaded historical forecasts from\n`{base_url_2}`\nsaved to\n{data_path}\n')


def read_forecast(file_path):
    '''
    Read and load wind forecast csv file as pandas dataframe.

    - file_path: location of wind forecast *.csv file
    '''
    df = pd.read_csv(file_path,header=3)
    df['Datetime'] = pd.to_datetime(df['Time']) # convert to datetime
    df.set_index('Datetime',inplace=True) # re-index
    df.index = df.index.tz_convert(None) # remove UTC
    del df['Time'] # delete "time" column
    return df
