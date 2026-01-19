from google.colab import files

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from scipy.stats import chi2_contingency
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder




uploaded = files.upload()




train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
submission = pd.read_csv("sample_submission.csv")



X = train.drop(columns=["fraud", "ID"])
y = train["fraud"]
X_test = test.drop(columns=["ID"])



num_cols = ['age_of_driver', 'safty_rating', 'annual_income', 'past_num_of_claims', 'liab_prct', 'claim_est_payout', 'age_of_vehicle', 'vehicle_price', 'vehicle_weight']
cat_cols = ['gender', 'marital_status', 'high_education_ind', 'address_change_ind', 'living_status', 'accident_site', 'witness_present_ind', 'channel', 'policy_report_filed_ind', 'vehicle_category', 'vehicle_color']



corr = train[num_cols + ['fraud']].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.show()



drop_num_cols = ['annual_income', 'liab_prct']
X = X.drop(drop_num_cols, axis=1)
X_test = X_test.drop(drop_num_cols, axis=1)

num_cols = ['age_of_driver', 'safty_rating', 'past_num_of_claims', 'claim_est_payout', 'age_of_vehicle', 'vehicle_price', 'vehicle_weight']



num_imp = SimpleImputer(strategy="median")
ss = StandardScaler()
X[num_cols] = num_imp.fit_transform(X[num_cols])
X_test[num_cols] = num_imp.transform(X_test[num_cols])
X[num_cols] = ss.fit_transform(X[num_cols])
X_test[num_cols] = ss.transform(X_test[num_cols])



for col in cat_cols:
    contingency = pd.crosstab(X[col], y)
    chi2, p, dof, expected = chi2_contingency(contingency)
    print(f"{col}: p-value = {p:.4f}")



drop_cat_cols = ['living_status', 'channel', 'vehicle_category', 'vehicle_color']
X = X.drop(drop_cat_cols, axis=1)
X_test = X_test.drop(drop_cat_cols, axis=1)

cat_cols = ['gender', 'marital_status', 'high_education_ind', 'address_change_ind', 'accident_site', 'witness_present_ind', 'policy_report_filed_ind']



cat_imp = SimpleImputer(strategy="most_frequent")
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
X[cat_cols] = cat_imp.fit_transform(X[cat_cols])
X_test[cat_cols] = cat_imp.transform(X_test[cat_cols])
X_cat_ohe = ohe.fit_transform(X[cat_cols])
X_test_cat_ohe = ohe.transform(X_test[cat_cols])

cat_ohe_cols = ohe.get_feature_names_out(cat_cols)

X_cat_ohe_df = pd.DataFrame(X_cat_ohe, columns=cat_ohe_cols, index=X.index)
X_test_cat_ohe_df = pd.DataFrame(X_test_cat_ohe, columns=cat_ohe_cols, index=X_test.index)
X = pd.concat([X, X_cat_ohe_df], axis=1)
X_test = pd.concat([X_test, X_test_cat_ohe_df], axis=1)
X = X.drop(cat_cols, axis=1)
X_test = X_test.drop(cat_cols, axis=1)



day_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
X['claim_day_of_week_map'] = X['claim_day_of_week'].map(day_map)
X_test['claim_day_of_week_map'] = X_test['claim_day_of_week'].map(day_map)
X['claim_day_of_week_sin'] = np.sin(2*np.pi*X['claim_day_of_week_map']/7)
X['claim_day_of_week_cos'] = np.cos(2*np.pi*X['claim_day_of_week_map']/7)
X_test['claim_day_of_week_sin'] = np.sin(2*np.pi*X_test['claim_day_of_week_map']/7)
X_test['claim_day_of_week_cos'] = np.cos(2*np.pi*X_test['claim_day_of_week_map']/7)



year_map = {2016: 0, 2017: 1}
X['year'] = X['year'].map(year_map)
X_test['year'] = X_test['year'].map(year_map)



month_map = {'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6, 'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12}
X['month_num'] = X['month'].map(month_map)
X_test['month_num'] = X_test['month'].map(month_map)
X['month_sin'] = np.sin(2*np.pi*X['month_num']/12)
X['month_cos'] = np.cos(2*np.pi*X['month_num']/12)
X_test['month_sin'] = np.sin(2*np.pi*X_test['month_num']/12)
X_test['month_cos'] = np.cos(2*np.pi*X_test['month_num']/12)



X['day_sin'] = np.sin(2*np.pi*X['day']/31)
X['day_cos'] = np.cos(2*np.pi*X['day']/31)
X_test['day_sin'] = np.sin(2*np.pi*X_test['day']/31)
X_test['day_cos'] = np.cos(2*np.pi*X_test['day']/31)



X = X.drop(['claim_day_of_week', 'claim_day_of_week_map', 'month', 'month_num', 'day'], axis=1)
X_test = X_test.drop(['claim_day_of_week', 'claim_day_of_week_map', 'month', 'month_num', 'day'], axis=1)



X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

smote = SMOTE(random_state=42)
X_train_overfit, y_train_overfit = smote.fit_resample(X_train, y_train)



lr = LogisticRegression(max_iter=2000, random_state=42)
lr.fit(X_train_overfit, y_train_overfit)

train_pred = lr.predict(X_train)
print(f"[Train] F1: {f1_score(y_train, train_pred):.4f}")

val_pred = lr.predict(X_val)
print(f"[Validation] F1: {f1_score(y_val, val_pred):.4f}")



rf = RandomForestClassifier(n_estimators=1000, max_depth=20, random_state=42)
rf.fit(X_train_overfit, y_train_overfit)

train_pred = rf.predict(X_train)
print(f"[Train] F1: {f1_score(y_train, train_pred):.4f}")

val_pred = rf.predict(X_val)
print(f"[Validation] F1: {f1_score(y_val, val_pred):.4f}")



lr.fit(X, y)
test_pred = lr.predict(X_test)

submission["fraud"] = test_pred
submission.to_csv("submission.csv", index=False)

files.download("submission.csv")