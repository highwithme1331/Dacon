#StandardScaler
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(data)
scaled_data = scaler.transform(data)



#Log Transform
train['col'] = np.log(train['col'])
test['col'] = np.log(test['col'])



#Train_Test_Split
from sklearn.model_selection import train_test_split 

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=False
)



#RMSE
from sklearn.metrics import mean_squared_error

rmse = np.sqrt(mean_squared_error(y_true, y_pred))



#MAE
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_true, y_pred)