#가중치 조합 탐색
from itertools import product

weights_range = np.arange(0.1, 0.9, 0.1)

best_weights = None
best_score = float('inf')

for weights in product(weights_range, repeat=3):
    if sum(weights) != 1:
        continue
    
    weighted_avg_preds = (weights[0]*rf_preds+weights[1]*gb_preds+weights[2]*et_preds)
    weighted_avg_score = root_mean_squared_error(y_valid, weighted_avg_preds)

    if weighted_avg_score < best_score:
        best_weights = weights
        best_score = weighted_avg_score



#검증점수 가중치
rf_score = root_mean_squared_error(y_valid, rf_preds)
gb_score = root_mean_squared_error(y_valid, gb_preds)
et_score = root_mean_squared_error(y_valid, et_preds)

rf_weight = 1/rf_score
gb_weight = 1/gb_score
et_weight = 1/et_score

total_weight = rf_weight+gb_weight+et_weight

rf_weight /= total_weight
gb_weight /= total_weight
et_weight /= total_weight

weighted_avg_preds = rf_weight*rf_preds+gb_weight*gb_preds+et_weight*et_preds
weighted_avg_score = root_mean_squared_error(y_valid, weighted_avg_preds)



#Seed Ensemble
all_predictions = []
test_predictions = []
seeds = [42, 77, 2024]

for seed in seeds:
    seed_et_model = ExtraTreesRegressor(n_estimators=20, random_state=seed)
    seed_et_model.fit(x_train, y_train)

    seed_et_pred = seed_et_model.predict(x_valid)

    all_predictions.append(seed_et_pred)
    
seed_pred = np.mean(all_predictions, axis=0)
rmse_seed_val = root_mean_squared_error(y_valid, seed_pred)



#KFold Ensemble
kf = KFold(n_splits=5, shuffle=True, random_state=42)

models = []
scores = []

for train_idx, valid_idx in kf.split(drop_train):
    x_train, x_valid = drop_train.iloc[train_idx], drop_train.iloc[valid_idx]
    y_train, y_valid = target.iloc[train_idx], target.iloc[valid_idx]

    model = ExtraTreesRegressor(n_estimators=20, random_state=42)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_valid)
    score = root_mean_squared_error(y_valid, y_pred)

    models.append(model)
    scores.append(score)



#Seed KFold Ensemble
seeds = [42, 77, 2024]

all_scores = []
all_models = [] 

for seed in seeds:
    fold_scores = []

    kf = KFold(n_splits=5, shuffle=True, random_state=seed)

    for train_idx, valid_idx in kf.split(drop_train):
        x_train, x_valid = drop_train.iloc[train_idx], drop_train.iloc[valid_idx]
        y_train, y_valid = target.iloc[train_idx], target.iloc[valid_idx]

        model = ExtraTreesRegressor(n_estimators=20, random_state=seed)
        model.fit(x_train, y_train)

        y_pred = model.predict(x_valid)
        score = root_mean_squared_error(y_valid, y_pred)

        all_models.append(model) 
        fold_scores.append(score)
    
    average_seed_score = np.mean(fold_scores)
    all_scores.append(average_seed_score)

overall_average_score = np.mean(all_scores)