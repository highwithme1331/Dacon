#seasonal_decompose
from statsmodels.tsa.seasonal import seasonal_decompose  

cyclic_period = 20

result = seasonal_decompose(df['col'], model='additive', period=cyclic_period) 


decomposition_multiplicative = seasonal_decompose(df['col'], model='additive')
decomposition_additive = seasonal_decompose(df['col'], model='multiplicative')  



#STL
from statsmodels.tsa.seasonal import STL

stl = STL(df['col'], period=20)
result = stl.fit()



#STL Feature
def generate_stl_features(data, window_size, target_col='col'):
    trend_values = []
    seasonal_values = []
    residual_values = []
    
    for i in range(window_size, len(data)):
        window_data = data[target_col].iloc[i-window_size:i]
        stl = STL(window_data, period=24)
        result = stl.fit()
        
        trend_values.append(result.trend.iloc[-1])
        seasonal_values.append(result.seasonal.iloc[-1])
        residual_values.append(result.resid.iloc[-1])
    
    result_df = data.iloc[window_size:].reset_index(drop=True)
    result_df['trend'] = trend_values
    result_df['seasonal'] = seasonal_values
    result_df['residual'] = residual_values
    
    return result_df


data_test_extended = pd.concat([data_train.tail(window_size), data_test], ignore_index=True)
data_test_stl_extended = generate_stl_features(data_test_extended, window_size=window_size, target_col='col’)
data_test_stl = data_test_stl_extended.reset_index(drop=True)



#Multiple Step Prediction
def create_target(data, target_col, n_steps):
    target_df = pd.DataFrame(index=data.index)

    for i in range(1, n_steps+1):
        col_name = f'{target_col}_t+{i}'
        target_df[col_name] = data[target_col].shift(-i)
    
    combined_df = pd.concat([data, target_df], axis=1)
    combined_df = combined_df.dropna().reset_index(drop=True)
    
    return combined_df

n_steps = 24

data_train_final = create_target(data_train_stl, 'col', n_steps)
data_test_final = create_target(data_test_stl, 'col', n_steps)



#Fourier Transform
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import numpy as np

fft_result = fft(df_signal['final_signal'].values)
fft_freq = fftfreq(len(df_signal), d=1/sampling_freq)
fft_magnitude = np.abs(fft_result)


filtered_fft_result = np.zeros_like(fft_result)
mask = fft_magnitude>threshold
filtered_fft_result[mask] = fft_result[mask]
reconstructed_signal = np.real(np.fft.ifft(filtered_fft_result))