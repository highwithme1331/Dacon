#LightGBM
from lightgbm import LGBMClassifier

target = train['target']
independent = train.drop(['index', 'target'], axis=1)

object_cols = [col for col in independent.columns if independent[col].dtype=='object']
independent[object_cols] = independent[object_cols].astype('category')


from lightgbm.callback import early_stopping, log_evaluation

tuning_lgbm = LGBMClassifier(n_estimators=300, max_depth=6, random_state=42)

early_stop = early_stopping(stopping_rounds=5)
log_eval = log_evaluation(period=20)

tuning_lgbm.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    eval_metric='multi_logloss',
    callbacks=[early_stop, log_eval]
)

pred = tuning_lgbm.predict(X_valid)
logloss_pred = tuning_lgbm.predict_proba(X_valid)