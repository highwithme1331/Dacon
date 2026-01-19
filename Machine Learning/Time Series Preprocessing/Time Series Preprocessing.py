#시계열 데이터 결측치 탐색 및 시각화
train['date_time'] = pd.to_datetime(train['date_time'])
train = train.set_index('date_time')

plt.figure(figsize=(8, 6))
ax = sns.heatmap(
    train[['A', 'B', 'C']].isnull(), 
    cbar=False, 
    cmap='viridis’
)



#시계열 데이터 결측치 처리
df['A_Interpolated'] = df['A'].interpolate(method='linear')

df['A_Fill'] = df['A'].bfill()



#시계열 데이터 Downsampling
weekly_data = df.resample('W').mean()
daily_data = df.resample('D').mean()



#시계열 데이터 Upsampling
minutely_data = df.resample('10min').interpolate()



#이동 평균을 사용한 시계열 데이터 평활화
num_window = 20
sma_col = f"SMA{num_window}"

daily_data[sma_col] = daily_data['target'].rolling(window=num_window).mean()



#지수 평균 이동을 사용한 시계열 데이터의 평활화
num_window = 20
ema_col = f"EMA{num_window}"

daily_data[ema_col] = daily_data[target'].ewm(span=num_window, adjust=False).mean()



#순환적 특성 표현(hour)
df['hour_sin'] = np.sin(2*np.pi*df['date_time'].dt.hour/24)
df['hour_cos'] = np.cos(2*np.pi*df['date_time'].dt.hour/24)