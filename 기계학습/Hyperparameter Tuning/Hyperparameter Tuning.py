#KFold
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    accuracy_scores.append(accuracy)



#Ridge
from sklearn.linear_model import Ridge

kf = KFold(n_splits=5, shuffle=True, random_state=42)

alpha_list = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]
scores_list = []

for alpha in alpha_list:
    ridge_model = Ridge(alpha=alpha)
    scores = cross_val_score(
        ridge_model, X_train, y_train, scoring='neg_mean_squared_error', cv=kf
    )
    scores_list.append(np.mean(scores))

best_score = max(scores_list)
optimal_alpha = alpha_list[np.argmax(scores_list)]



#Logistic Regression
kf = KFold(n_splits=5, shuffle=True, random_state=40)

c_list = [10e-7, 10e-6, 10e-5, 10e-4, 10e-3, 10e-2, 10e-1, 1, 10, 10**2]
scores_list = []

for c_val in c_list:
    logreg_model = LogisticRegression(C=c_val, solver='liblinear', random_state=42)
    scores = cross_val_score(logreg_model, X_train, y_train, scoring='accuracy', cv=kf)
    scores_list.append(np.mean(scores))

best_score = max(scores_list)
optimal_c = c_list[np.argmax(scores_list)]