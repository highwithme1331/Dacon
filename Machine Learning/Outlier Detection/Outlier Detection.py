#Z-Score
mean_train = train['col'].mean()
std_train = train['col'].std()

threshold = 3

train['z_score'] = (train['col']-mean_train)/std_train
train_no_outliers = train[train['z_score'].abs()<=threshold]



#Z-Score(scipy)
from scipy import stats

train['z_score'] = stats.zscore(train['col'])

threshold = 3

train_no_outliers = train[train['z_score'].abs()<=threshold]



#IQR
def out_iqr(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3-Q1
    lower_bound = Q1–1.5*IQR
    upper_bound = Q3+1.5*IQR
    return lower_bound, upper_bound

lower_bound, upper_bound = out_iqr(train['col'])

lower_outliers = train[train['col']<lower_bound]
upper_outliers = train[train['col']>upper_bound]

train_no_outliers_iqr = train[
    (train['col']>=lower_bound) &
    (train['col']<=upper_bound)
]



#IQR(percentile)
Q1 = np.percentile(train['col'], 25)
Q3 = np.percentile(train['col'], 75)

IQR = Q3-Q1
lower_bound = Q1-1.5*IQR
upper_bound = Q3+1.5*IQR

train_no_outliers_iqr_np = train[
    (train[col]>=lower_bound) &
    (train[col]<=upper_bound)
]



#DBSCAN
from sklearn.cluster import DBSCAN

numeric_columns = ['A', 'B', ‘C’]
data_numeric = train[numeric_columns]

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_numeric)

db = DBSCAN(eps=0.5, min_samples=5)
labels = db.fit_predict(data_scaled)

train['clusters'] = labels
sample_no_outliers = train[train['clusters']!=-1]



#LOF
from sklearn.neighbors import LocalOutlierFactor

numeric_columns = ['A', 'B', ‘C’]
data_numeric = train[numeric_columns]

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_numeric)

lof = LocalOutlierFactor(n_neighbors=50, contamination='auto')
labels = lof.fit_predict(data_scaled)

train['outliers_lof'] = labels
sample_no_outliers = train[train['outliers_lof']!=-1]