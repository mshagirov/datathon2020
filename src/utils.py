import numpy as np

# metrics
def mse(x1,x2):
    '''Compute Mean Squared Error (MSE loss) between numpy arrays x1 and x2'''
    return np.mean( (x2-x1)**2 )

def rmse(x1,x2):
    '''Compute Root Mean Squared Error (RMSE loss) between numpy arrays x1 and x2'''
    return np.sqrt( np.mean((x2-x1)**2) )

def mae(x1,x2):
    '''Compute Mean Absolute Error (MAE loss) between numpy arrays x1 and x2'''
    return np.mean( np.absolute(x2-x1) )


def lagged_xcorr(x1,x2):
    '''"Lagged correlation",
    
    ouput:
    - lags, xcorr : lag times, and corresponding xcorr values
    
    xcorr val-s are devided by x1.shape[0] (produces average), and np.std(x1)*np.std(x2) (scale to normalize xcorr)
    '''
    assert x1.shape[0]==x2.shape[0]
    xcorr_scale = np.std(x1)*np.std(x2)
    # "full" xcorr will be "2 * x1.shape[0] -1", middle is lag==0
    xcorr = np.correlate(x1,x2,mode='full') /xcorr_scale /x1.shape[0]
    
    lags = np.concatenate(( np.arange(-(x1.shape[0]-1),1), np.arange(1,x1.shape[0]) ) )
    return lags, xcorr

# persistence
def persistence_loss(x,start_time=0,lead_time=5,metric='mse'):
    '''Compute loss for persistence with `T+lead_time`, i.e. RMSE(Y(T+0),Y(T+lead_time))

    - metric: one of ['mse','rmse', 'mae','none'], when 'none' returns Y(T+0) and Y(T+lead_time) values.
    
    - start_time = window_size - 1 , is the index of first Y(T+0)
    - must satisfy: "x.shape[0]-lead_time">start_time>=0
    '''
    Y0 = x[start_time:x.shape[0]-lead_time]
    Ylead = x[start_time+lead_time:]
    assert Y0.shape[0]==Ylead.shape[0]
    assert metric in ['mse','rmse', 'mae','none']
    if metric=='mse':
        return mse(Y0,Ylead)
    elif metric=='rmse':
        return rmse(Y0,Ylead)
    elif metric=='mae':
        return mae(Y0,Ylead)
    else:
        return Y0,Ylead


