#RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier

X, y = iris.data, iris.target 

model = RandomForestClassifier(random_state=42) 
model.fit(X, y)

new_data = [[5.1, 3.5, 1.4, 0.2]]
prediction = model.predict(new_data)



#Evaluation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy_RF = accuracy_score(y_val, y_pred)
precision_RF = precision_score(y_val, y_pred, average='macro')
recall_RF = recall_score(y_val, y_pred, average='macro')
f1_score_RF = f1_score(y_val, y_pred, average='macro')