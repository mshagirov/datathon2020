# # Set to FALSE when deploying!!! # #
TESTING = False

# # # # # # # # # # # # # # # # # # #

import os, glob, pickle
import urllib.request
import torch, numpy as np

import os, pandas as pd, numpy as np
import time, datetime

if torch.cuda.is_available():
#     device = torch.device('cuda')
    device = torch.device('cuda:3')
else:
    device = torch.device("cpu")
print(f'Available device: {device}')


#For running in JupyterHub:
if os.path.basename(os.getcwd())!='P003':
    print('Not in /P003 folder, changing directory to P003')
    lib_path = os.path.expanduser(os.path.relpath('~/images/codesDIR/datathon2020/P003'))
    os.chdir(lib_path)
    
from src import datautils
from src.nnets import diffmlp, dense

# # # # # # # # # # # # # # # # # # #
#             CONSTANTS             #
# # # # # # # # # # # # # # # # # # #
window_size = 40
lead_time = 18

# Normalisation:
# ENERGY :
shift_ = 18000.0 # e.g. mean , lookup from stats, you can round it up
scale_ = 40000.0 # e.g. 2xS.D., lookup from stats
# x_norm = (x-shift_)/scale_

# WIND SPEED:
wind_scale_ = 8.0 # half of max speed (wind vector)

# location of networks
model_dir = os.path.relpath('./best_models')
# get all models that match the pattern:
model_filenames = glob.glob(os.path.join(model_dir,'27Jul2020_1023*'))

# Just in case
deployment_end_time  = pd.to_datetime('2020-12-29 10:00:00')

MIN_ENERGY =  5000 #minimum energy
print(f'MIN_ENERGY is set to :{MIN_ENERGY}')

# # # # # # # # # # # # # # # # # # #
#        HELPER FUNCTIONS           #
# # # # # # # # # # # # # # # # # # #
# my server's offset time:
time_offset = pd.Timedelta(minutes=0,seconds=0) # correction to cpu time
# jhub offset--> pd.Timedelta(minutes=6,seconds=-9)
if TESTING:
    time_offset = pd.Timedelta(hours=-10,minutes=20,seconds=9)

def utc_now(time_delta = time_offset):
    t = datetime.datetime.utcnow()
    utc_time = pd.to_datetime('{}-{:02d}-{:02d} {:02d}:{:02d}:{:02d}'.format(
        t.year,t.month,t.day, t.hour,t.minute,t.second))+time_delta
    return utc_time

def need_time():
    '''Current time needed for Energy kWh readings'''
    curr_time_utc = utc_now()
    curr_minutes = curr_time_utc.minute
    
    if (curr_minutes<10) or (curr_minutes>29):
        return curr_time_utc.round('H')-pd.Timedelta(hours=1)
    return curr_time_utc.round('H')


def send_value2url(val):
    html=b''
    passwrd = '961993551'
    url_add = f'http://3.1.52.222/submit/pred?pwd={passwrd}&value={val}'
    try:
        with urllib.request.urlopen(url_add) as response:
            html = response.read()
    except urllib.request.HTTPError:
        html = b'server is down'
        print(utc_now(),f' (for {need_time()}):',html.decode(),url_add)
    return html


def read_wind_forecasts_w_range(wind_speed_range):
    dfs = []
    for model_n in range(1,3,1):
        for farm_k in range(len(datautils.locations)):
            dfs.append(
                datautils.readlocation_as_vec( model_n,farm_k, wind_speed_range).interpolate(
                    method='linear').reindex(wind_speed_range) )
    # concatenate along axis 1 using datetime as reference
    # then take average of two forecasts
    return datautils.average_forecast_models( pd.concat(dfs,axis=1) )


modellist = []
for fname in model_filenames:
    with open(fname, 'rb') as f:
        print(f'Loading {fname}')
        modellist.append(pickle.load(f))

for net in modellist:
    net.to(device)

def predictT18(x,y_0,nets=modellist):
    results = []
    y_0 = torch.from_numpy(y_0.astype(np.float32)).to(device)
    x = torch.from_numpy(x.astype(np.float32)).to(device)
    with torch.no_grad():
        for net in modellist:
            net.eval()
            results.append(net(x,y_0).detach().cpu().numpy())
    return results



def iter_predict18(energy_df, latest_energy_time_):
    tau_fill = pd.date_range(latest_energy_time_+pd.Timedelta(hours=1), need_time(), freq='H')
    
    if tau_fill.shape[0]>39:
        # if missing more than 39 hours just return 0
        print(f'\nToo many missing values for energy (missing {tau_fill.shape[0]} recent samples.)\n')
        return 0
    # time range for Energy
    eng_fill_range = pd.date_range(
        tau_fill[0] - pd.Timedelta(hours=window_size-1+18), energy_df.index[-1], freq='H')
    # normalised energy (copy)
    eng_df_norm = ( energy_df.loc[eng_fill_range].copy(deep=True)- shift_)/scale_
    
    # time range for wind speed
    wind_speed_range = pd.date_range(eng_fill_range[0],
                                     need_time()+pd.Timedelta(hours=18),freq='H')
    wind_df_norm = read_wind_forecasts_w_range(wind_speed_range)/wind_scale_
    
    # fill in missing values
    for t in tau_fill:
        need_range = pd.date_range(t-pd.Timedelta(hours=window_size-1+18),
                                   t-pd.Timedelta(hours=18),freq='H')
        Y_fill = eng_df_norm.loc[need_range].values.reshape(1,-1)
        Y0_fill = Y_fill[:,:1]
        Ydiff_fill = Y_fill[:,0:-1] - Y_fill[:,1:]
        # wind needs T+18 forecast
        need_range_wind = pd.date_range(need_range[0],
                                        need_range[-1]+pd.Timedelta(hours=18),freq='H')
        
        X_df = wind_df_norm.loc[need_range_wind].values
        X_norm = np.sqrt(X_df[:,0::2]**2 + X_df[:,1::2]**2)# wind speed from wind vectors
        X_windows = []
        for l in range(X_norm.shape[1]):
            X_windows.append(
                datautils.windowed_diff_data(X_norm[:,l], lead_time=lead_time, window_size=window_size))
        
        X_fill = [Ydiff_fill]
        X_fill.extend(X_windows)
        X_fill = np.concatenate(X_fill, axis=1)
        eng_df_norm.loc[t] = np.mean(predictT18(X_fill,Y0_fill))
    
    # Predict T+18
    # Y(T+0) and diff-s
    need_range = pd.date_range(tau_fill[-1]-pd.Timedelta(hours=window_size-1),tau_fill[-1],freq='H')
    Y = eng_df_norm.loc[need_range].values.reshape(1,-1)
    Y0 = Y[:,:1]
    Ydiff = Y[:,0:-1] - Y[:,1:]
    
    # X(T+18), X(T+0), and diff-s 
    # wind needs T+18 forecast
    need_range_wind = pd.date_range(need_range[0], need_range[-1]+pd.Timedelta(hours=18),freq='H')
    X_df = wind_df_norm.loc[need_range_wind].values
    X_norm = np.sqrt(X_df[:,0::2]**2 + X_df[:,1::2]**2)# wind speed from wind vectors
    X_windows = []
    for l in range(X_norm.shape[1]):
        X_windows.append(
            datautils.windowed_diff_data(X_norm[:,l], lead_time=lead_time, window_size=window_size))
    
    Xdeploy = [Ydiff]
    Xdeploy.extend(X_windows)
    Xdeploy = np.concatenate(Xdeploy, axis=1)
    
    # Predict using 5 best models, and de-normalise, take mean for all predictions:
    Y_pred = np.mean(np.array(predictT18(Xdeploy,Y0))*scale_ +shift_)
    # Set 0kWh as min prediction, and convert to integer
    Y_pred = np.maximum(0,int(Y_pred))
    
    return Y_pred


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
print('\n'+'- - '*5+
      f'\nDeployment > {"Testing" if TESTING else " ON  <"}\n'+
      '- - '*5+'\n')
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # DEPLOYMENT LOOP # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
time_left = 0
while deployment_end_time>utc_now():
    # sleep until 50min
    curr_utc_time = utc_now()
    min_now = curr_utc_time.minute + curr_utc_time.second/60
    if (min_now<50) and (min_now>10):
        # deploy time 1: wait for HH:51min
        time_left = (50-min_now)*60 # in seconds
    elif (min_now>=50):
        # deploy time 2: wait for HH:00min
        time_left = (60-min_now)*60 # in seconds
    elif min_now<=9:
        #deploy time 3: wait for HH:09min
        time_left = (9-min_now)*60 # in seconds
    elif (min_now>9) and (min_now<50):
        # deploy time 1: wait for HH:51min
        time_left = (50-min_now)*60 # in seconds
    if (curr_utc_time.minute==0) and (curr_utc_time.second<15):
        time_left = 0.0
    print(f'\nSleep {time_left/60:2.0f}mins\n')
    
    if not TESTING:
        time.sleep(abs(time_left))

    t0 = utc_now()
    # Download New Forecasts:
    datautils.download_forecasts_all()

    # AI4Impact source:
    datautils.download_energy_latest()
    energy_1 = datautils.read_ai4impact_energy('datasets/energy-ile-de-france.csv')

    # RTE source:
    #RTE_file_path = datautils.download_raw_from_RTE('real_time',return_filelist=True)
    #energy_2=datautils.read_RTE_as_kwh(RTE_file_path[0],convert2UTC=True)

    print(f'---\nTime elapsed: {utc_now()-t0}\n---\n')

    # Select latest energy source
    #enrg_src = np.argmax([energy_1.index[-1], energy_2.index[-1]])
    energy_df  = energy_1 # energy_1 if enrg_src==0 else energy_2
    energy_date_range = pd.date_range(energy_df.index[0],energy_df.index[-1],freq='H')


    need_range = pd.date_range(need_time()-pd.Timedelta(hours=window_size-1),need_time(),freq='H')


    if energy_df[energy_df.index==need_time()].values.shape[0]==0:
        print(f'\nUsing iterative method: latest {energy_date_range[-1]} (need {need_time()})\n')
        Y_pred = iter_predict18(energy_df, energy_date_range[-1])
        pred_method = f'iterative ({energy_date_range[-1]})'
    else:
        print(f'\nComputing T+18 forecast directly (latest {energy_date_range[-1]}, need {need_time()})\n')
        # we need this range of dates for features
        oldest_time = need_time()-pd.Timedelta(hours=window_size-1)
        wind_speed_latest_time = need_time()+pd.Timedelta(hours=18)
        wind_speed_range = pd.date_range(oldest_time,wind_speed_latest_time,freq='H')

        wind_df = read_wind_forecasts_w_range(wind_speed_range)

        wind_norm = wind_df.values/wind_scale_
        wind_speeds = np.sqrt(wind_norm[:,0::2]**2 + wind_norm[:,1::2]**2)# wind speed from wind vectors

        X_norm_diffwindows = []
        for l in range(wind_speeds.shape[1]):
            X_norm_diffwindows.append(
                datautils.windowed_diff_data(wind_speeds[:,l], lead_time=lead_time, window_size=window_size))
        # print for debugging
        # print('\nX(t+18),X(T+0),X(T+0)-X(T-1),...,X(T-window_size+2)-X(T-window_size+1):\n',
        #       [l.shape for l in X_norm_diffwindows])
        Y = (energy_df.loc[need_range].values[::-1,0].reshape(1,-1) - shift_)/scale_
        Y0 = Y[:,:1]
        Ydiff = Y[:,0:-1] - Y[:,1:]
        Xdeploy = [Ydiff]
        Xdeploy.extend(X_norm_diffwindows)
        Xdeploy = np.concatenate(Xdeploy, axis=1)
        # Predict using 5 best models, and de-normalise, take mean for all predictions:
        Y_pred = np.mean(np.array(predictT18(Xdeploy,Y0))*scale_ +shift_)
        # Set 0kWh as min prediction, and convert to integer
        Y_pred = np.maximum(0,int(Y_pred))
        pred_method = f'direct ({energy_date_range[-1]})'
    
    curr_utc_time = utc_now()
    curr_mins = curr_utc_time.minute +curr_utc_time.second/60
    if (curr_mins<50) and (curr_mins>10):
        continue
    
    Y_pred = int(np.maximum(MIN_ENERGY,Y_pred))
    
    
    url_response = send_value2url(Y_pred)
    print(url_response.decode())
    print(f'---\n {utc_now()}: Time elapsed: {utc_now()-t0}\n---\n')
    
    with open('rt_predictions.txt', 'a') as the_file:
        the_file.write(f'{utc_now()}: {Y_pred} (for {need_time()}, {pred_method}) {url_response.decode()}\n')

