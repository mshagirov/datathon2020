import numpy as np
import urllib.request, os, zipfile, pandas as pd

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
    '''For 1d array Y, prepare windowed data with a given window size with differences
    - 1st and 2nd columns:`[Y(T+lead_time), Y(T+0),]
    - 3rd and subsequent columns are differences:
    `[, Y(T+0)-Y(T-1), Y(T-1)-Y(T-2),..., Y(T-window_size+2)-Y(T-window_size+1)]`
    '''
    X = windowed_data(x_norm, lead_time=lead_time, window_size=window_size)
    X[:,2:] = X[:,1:-1]-X[:,2:]
    # concatenate Y(T+lead_time), Y(T+0), differences
    return X


def windowed_diff_with_step(Y,lead_time=5, window_size=6, h=5):
    '''For 1d array Y, computes differences between elements Y[i]-Y[i-h], where h is a step size.
    Generally, h is chosen to be equal to the lead time: `h=lead_time`.

    Arg-s:
    - Y: input (1d array, e.g. time series data), this should be normalised e.g. x_norm.
    - lead_time: lead time for Y(t+lead_time) you are predicting using data from Y(t+0) and before.
    - window_size : window size, features are computed from a window with size "window_size".
    This window includes [Y(t+0),...., Y(t-window_size+1)].
    - h : step size for differences Y[i]-Y[i-h].

    Returns an array with:
    - len(Y) - (`window_size`+lead_time-1) rows, and
    - (`window_size`-h)+2 columns, where 1st col. is target Y[t+lead_time],
    2nd col is current Y[t+0], and (`window_size`-h)di fferences with step h:
    [ Y[t+0]-Y[t-h], Y[t-1]-Y[(t-1)-h], ..., Y[(t-(window_size-1-h)]-Y[t-window_size+1]  ]
    '''
    X = windowed_data(Y, lead_time=lead_time, window_size=window_size)
    Xout = [X[:,:2], X[:,1:(window_size+1-h)]-X[:,1+h:]] # [Ylead, Y0, Y[t+0]-Y[t-h],Y[t-1]-Y[t-h-1],... ]
    return np.concatenate(Xout,axis=1)


def windowed_momentum_force(Y,lead_time=5, window_size=6, h=5):
    '''For 1d array Y, computes higher-order differences between elements Y[i]-Y[i-h], where h is a step size.
    Generally, h is chosen to be equal to the lead time: `h=lead_time`.

    Arg-s:
    - Y: input (1d array, e.g. time series data), this should be normalised e.g. x_norm.
    - lead_time: lead time for Y(t+lead_time) you are predicting using data from Y(t+0) and before.
    - window_size : window size, features are computed from a window with size "window_size".
    This window includes [Y(t+0),...., Y(t-window_size+1)].
    - h : step size for differences Y[i]-Y[i-h].

    Returns an array with:
    - len(Y) - (`window_size`+lead_time-1) rows, and
    - First 2 columns: [ Y[t+lead_time],  Y[t+0] ]
    - Next (`window_size`-h) col-s are differences dY[t] with step h:
    [ Y[t+0]-Y[t-h], Y[t-1]-Y[(t-1)-h], ..., Y[(t-(window_size-1-h)]-Y[t-window_size+1]  ]
    - Next (`window_size`-2*h) col-s are momentums ddY[t], differences of differences with step h:
    [ dY[t+0] - dY[t-h], ..., dY[(t-(window_size-1-2h)] - dY[(t-(window_size-1-h)]]
    - The last (`window_size`-3*h) col-s are forces==differences of momentums with step h:
    [ ddY[t+0] - ddY[t-h], ..., ddY[(t-(window_size-1-3h)] - ddY[(t-(window_size-1-2h)]] or
    [ (Y[t+0]-Y[t-h]) - 2*(Y[t-h]-Y[t-2h])+ ((Y[t-2h]-Y[t-3h])) ), ...]
    '''
    X = windowed_data(Y, lead_time=lead_time, window_size=window_size)
    YleadY0 = X[:,:2] # [Ylead, Y0]
    diffX = X[:,1:(window_size+1-h)]-X[:,1+h:] # [ Y[t+0]-Y[t-h],Y[t-1]-Y[t-h-1],... ]
    momentumX = diffX[:,:diffX.shape[1]-h] - diffX[:,h:] # [ (Y[t+0]-Y[t-h]) - ((Y[t-h]-Y[t-2h]))]
    forceX = momentumX[:,:momentumX.shape[1]-h] - momentumX[:,h:] # [ (Y[t+0]-Y[t-h]) - 2*(Y[t-h]-Y[t-2h])+ ((Y[t-2h]-Y[t-3h])) )]
    return np.concatenate([YleadY0, diffX, momentumX, forceX], axis=1)


def download_energy_latest(file_url_='https://ai4impact.org/P003/historical/energy-ile-de-france.csv',
file_path='datasets/energy-ile-de-france.csv',return_filepath=False):
    '''
    Download historical and the latest RTE energy readings from
    `https://ai4impact.org/P003/historical/energy-ile-de-france.csv`.
    Save all files to `./datasets/`, formatted to hourly kWh readings.

    file_url_ : url to download from, default is the url above
    file_path : file name and location for the downloaded file.
    return_filepath: return location of saved file (==file_path)
    '''
    # Download latest wind forecasts
    file_name,_ = urllib.request.urlretrieve(file_url_,filename=file_path)
    assert file_name==file_path
    print(f'Downloaded from:\n`{file_url_}`\nsaved to:\n{file_name}')
    if return_filepath:
        return file_name


def read_ai4impact_energy(file_path):
    '''
    Read energy (in kWh) readings from "energy-ile-de-france.csv".

    You NEED TO update the leatest readings using `download_energy_latest()`,
    and then use this function to read the downloaded `*.csv`.
    '''
    df = pd.read_csv(file_path,header=None,index_col=0)
    df.index.rename('Datetime',inplace=True)
    df.index=pd.to_datetime(df.index)
    df.rename(columns={1:'Energy(kWh)'},inplace=True)
    return df

# - For Ile-de-France, RTE links for each year/period, UNITS: MW (power):
#     - **2017** ("Definitive data for the year 2017"):<br>
#     `https://eco2mix.rte-france.com/download/eco2mix/eCO2mix_RTE_Ile-de-France_Annuel-Definitif_2017.zip`
#     - **2018** ("Definitive data for the year 2018"):<br>
#     `https://eco2mix.rte-france.com/download/eco2mix/eCO2mix_RTE_Ile-de-France_Annuel-Definitif_2018.zip`
#     - **01/01/2019-31/05/2020** ("Current consolidated data"):<br>
#     `https://eco2mix.rte-france.com/download/eco2mix/eCO2mix_RTE_Ile-de-France_En-cours-Consolide.zip`
#     - **01/06/2020-09/07/2020** ("Current real-time data"):<br>
#     `https://eco2mix.rte-france.com/download/eco2mix/eCO2mix_RTE_Ile-de-France_En-cours-TR.zip`
# for 2019_May2020 data need to remove extra COLUMNS (each row has different #col-s)
wind_energy_urls = {"2017":"https://eco2mix.rte-france.com/download/eco2mix/eCO2mix_RTE_Ile-de-France_Annuel-Definitif_2017.zip",
                    "2018":"https://eco2mix.rte-france.com/download/eco2mix/eCO2mix_RTE_Ile-de-France_Annuel-Definitif_2018.zip",
                    "2019_May2020":"https://eco2mix.rte-france.com/download/eco2mix/eCO2mix_RTE_Ile-de-France_En-cours-Consolide.zip",
                    "real_time":"https://eco2mix.rte-france.com/download/eco2mix/eCO2mix_RTE_Ile-de-France_En-cours-TR.zip"}

def download_raw_from_RTE(RTE_data_name='real_time',data_path='datasets',keep_zip=False,
return_filelist=False):
    '''
    Download raw RTE datasets for all types of energy production (zip file),
    and unzip it (xls file ending , but it's a csv file). Readings are power (MW),
    i.e. rate of energy production.
    - RTE_data_name: one of ['real_time','2017','2018','2019_May2020'], {default:'real_time'}.
    These are keys for dict that stores urls, e.g. `datautils.wind_energy_urls['real_time']` .
    - data_path: directory to save raw dataset {default: '/datasets'}.
    - keep_zip: if False deletes the downloaded zip file after unzipping.
    - return_filelist: if True returns list of (unzipped) downloaded files (list of paths, str).

    **for 2019_May2020 data** you will need to remove extra COLUMNS (each row has different #col-s),
    before reading with pandas.
    '''
    # download the current data +/-15mins sometimes even later, ~24 hrs delay
    zip_file_name = wind_energy_urls[RTE_data_name].split('/')[-1]
    print(f'Downloading\n\"{zip_file_name}\" from\n`{wind_energy_urls[RTE_data_name]}`\n')

    file_name = os.path.join(data_path,zip_file_name)
    zip_file_name,_ = urllib.request.urlretrieve(wind_energy_urls[RTE_data_name],filename=file_name)
    # unzip
    with zipfile.ZipFile(zip_file_name, 'r') as zip_ref:
        print(f'Extracting: {zip_file_name}')
        zip_filelist=[]
        for fn in zip_ref.filelist:
            print(f'{fn.filename} --> as \"{os.path.join(data_path,fn.filename)}\"')
            if return_filelist:
                zip_filelist.append(os.path.join(data_path,fn.filename))
        # file modification date and time: zip_ref.filelist[0].date_time
        print(5*'- ')
        zip_ref.printdir()
        zip_ref.extractall(data_path)
    if not keep_zip:
        if os.path.exists(zip_file_name):
            print(+5*'- '+f'\nDeleting\n{zip_file_name}\n')
            os.remove(zip_file_name)
    if return_filelist:
        return zip_filelist

def read_RTE_as_kwh(RTE_file_path,convert2UTC=True):
    '''
    Read RTE `*.csv/*.xls` file from "RTE_file_path" and convert power (MW) to average energies (kWh),
    by computing hourly average power (~MWh) then x1000. Converts Paris time to UTC when convert2UTC==True.

    Returns pandas dataframe (also removes NA values), freq='H', energy units in kWh.

    - convert2UTC: convert datetime from Paris time to UTC, this should be set to True when reading real-time
    '''
    df = pd.read_csv(RTE_file_path,sep='\t',header=0,encoding='latin',index_col=False,
                     parse_dates=[['Date', 'Heures']], na_values=['-','ND',' ','','DC'])[['Date_Heures','Eolien']]
    # The "DC" label which can appear for certain power generation sectors in some regions, means "confidential data"
    # The "ND" label means "unavailable data": the data do not exist for the requested period.
    df = df.iloc[:-1] # remove the footer with nan for Date and time (disclaimer line)
    # set index as Datetime
    df['Datetime'] = pd.to_datetime(df['Date_Heures'],dayfirst=True)
    df.set_index('Datetime',inplace=True)

    # real time data is in Paris time zone (summer):
    if convert2UTC:
        # convert real-time data time stamp Paris-->UTC time zone
        df.index=df.index.tz_localize('Europe/Paris').tz_convert(None)

    # re-sample for 1H interval and take its mean to find average power over 1H period
    # (equivalent to integrating over 1h periods to get average energy produced over 1H)
    df = df.resample('H').mean()

    # convert units
    df = df*1000 # MWh to kWh

    # remove NaNs
    df = df[np.logical_not(df['Eolien'].isna())]
    df.rename(columns={'Eolien':'Energy(kWh)'},inplace=True)
    return df


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

def join_model_historical(fname_hist,fname_model,energy_df_index):
    '''read historical and new forecasts and join them in time,
    new index will be from earliest to latest time, min and max of both
    forecasts and energy times.'''
    
    df_hist = read_forecast(fname_hist)
    df_modl = read_forecast(fname_model)
    df_modl = df_modl.loc[df_modl.index>df_hist.index.max()]
    df=pd.concat([df_hist,df_modl], axis=0,sort=False)
    
    date_range = pd.date_range(min([energy_df_index.min(),df.index.min()]),
                               max([energy_df_index.max(),df.index.max()]),freq='H')
    return df.reindex(date_range)


def direction2vector(df,df_name):
    ''' Convert Speed and Direction(angle) into 2D vector using [sin,cos]
    df should have columns "Speed(m/s)" and "Direction (deg N)",
    names for new columns will start with "df_name"'''
    dir_in_radians = np.deg2rad( df['Direction (deg N)'].values )

    newdf = pd.DataFrame({df_name+'_sin':df['Speed(m/s)'].values*np.sin(dir_in_radians),
                          df_name+'_cos':df['Speed(m/s)'].values*np.cos(dir_in_radians)},
                         index=df.index)
    return newdf


def readlocation_as_vec(model_n,loc_k, energy_df_index):
    '''model_n : 1 or 2 (model type), loc_k: 0 to 7 (index for datautils.locations)'''
    df_name = f"model{model_n}_{locations[loc_k]}"
    df = join_model_historical(f"./datasets/historical{model_n}/{locations[loc_k]}{'-b' if model_n==2 else ''}.csv",
                               f"./datasets/model{model_n}/{locations[loc_k]}{'-b' if model_n==2 else ''}.csv" ,
                              energy_df_index)
    return direction2vector(df,df_name)


def average_forecast_models(forecast_df):
    '''Average wind forecast vectors (sin,cos) from to forecast models.
    - forecast_df must have columns "model1_{farm}_sin" and  "model2_{farm}_cos"
    where "farm" is one of the locations (one from datautils.locations).'''
    dfs_avgmodels={}
    for farm in locations:
        dfs_avgmodels[f"{farm}_sin"] = (forecast_df[f"model1_{farm}_sin"] + forecast_df[f"model2_{farm}_sin"])/2
        dfs_avgmodels[f"{farm}_cos"] = (forecast_df[f"model1_{farm}_cos"] + forecast_df[f"model2_{farm}_cos"])/2
    return pd.DataFrame(dfs_avgmodels)
