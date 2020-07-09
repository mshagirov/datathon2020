import numpy as np

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
