#Voting
from sklearn.ensemble import VotingClassifier

rf_model = RandomForestClassifier(n_estimators=10)
gb_model = GradientBoostingClassifier(n_estimators=10)
et_model = ExtraTreesClassifier(n_estimators=10)

voting_clf = VotingClassifier(
    estimators=[('rf', rf_model), ('gb', gb_model), ('et', et_model)],
    voting='hard',
    weights=[1, 2, 3]
)

voting_clf.fit(x_train, y_train)
y_pred = voting_clf.predict(x_test)



#Stacking
from sklearn.ensemble import StackingClassifier

estimators = [
    ('rf', RandomForestClassifier(n_estimators=10)),
    ('gb', GradientBoostingClassifier(n_estimators=10)),
    ('et', ExtraTreesClassifier(n_estimators=10))
]

meta_model = RandomForestClassifier(n_estimators=10, random_state=42)

stacked_model = StackingClassifier(
    estimators=estimators,
    final_estimator=meta_model,
    cv=3
)

stacked_model.fit(train_x, train_y)
stack_pred = stacked_model.predict(x_test)