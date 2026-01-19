#RandomOverSampler
from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)



#SMOTE
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X_train, y_train)



#RandomUnderSampler
from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(random_state=42)
X_under, y_under = rus.fit_resample(X_train, y_train)



#NearMiss
from imblearn.under_sampling import NearMiss

nm1 = NearMiss(version=1) 
nm2 = NearMiss(version=2)
nm3 = NearMiss(version=3)

X_nm1, y_nm1 = nm1.fit_resample(X_train, y_train)
X_nm2, y_nm2 = nm2.fit_resample(X_train, y_train)
X_nm3, y_nm3 = nm3.fit_resample(X_train, y_train)