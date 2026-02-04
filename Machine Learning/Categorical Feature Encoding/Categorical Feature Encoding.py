#Label Encoding
from sklearn.preprocessing import LabelEncoder

label_encoders = {}

for col in categorical_var:
    le_train = LabelEncoder()

    train[col] = le_train.fit_transform(train[col])
    label_encoders[col] = le_train

for col in categorical_var:
    try:
        test[col] = label_encoders[col].transform(test[col])
    
    except Exception as e:
        print("Error")



#학습 데이터에 없는 범주 처리
for le in label_encoders.values():
    le.classes_ = np.append(le.classes_, 'Other')

for col in categorical_var:
    test[col] = test[col].apply(
        lambda x: x if x in label_encoders[col].classes_ else 'Other’
    )

    test[col] = label_encoders[col].transform(test[col])



#Direct Mapping
direct_map = {'A':0, 'B':1, 'C':2} 

train['col'] = train['col'].replace(direct_map)

test['col'] = test['col'].replace(direct_map)



#Ordinal Encoder
from sklearn.preprocessing import OrdinalEncoder

encoder_train = OrdinalEncoder(categories=[['A', 'B', 'C']])

train['col'] = encoder_train.fit_transform(train[['col']]).astype(int)

test['col'] = encoder_train.transform(test[['col']]).astype(int)



#One-Hot Encoding
from sklearn.preprocessing import OneHotEncoder  

encoder = OneHotEncoder(sparse_output=False)
train_encoded = encoder.fit_transform(train[['A', 'B', ‘C’]])

encoded_columns = encoder.get_feature_names_out(['A', 'B', ‘C’])
train_encoded_df = pd.DataFrame(train_encoded, columns=encoded_columns)

train = pd.concat([train, train_encoded_df], axis=1)
train = train.drop(['A', 'B', ‘C’], axis=1)

try :
    test_encoded = encoder.transform(test[['A', 'B', ‘C’]])
    test_encoded_df = pd.DataFrame(test_encoded, columns=encoded_columns)
    test = pd.concat([test, test_encoded_df], axis=1)
    test = test.drop(['A', 'B', ‘C’], axis=1)



#Ignore
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')



#Drop
encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore’)



#Binary Encoding
import category_encoders as ce

train['col'].fillna('Unknown', inplace=True)

encoder = ce.BinaryEncoder(cols=['col'])
train_encoded = encoder.fit_transform(train['col'])

encoded_columns = train_encoded.columns
train_encoded_df = pd.DataFrame(train_encoded, columns=encoded_columns)

train = pd.concat([train, train_encoded_df], axis=1)
train = train.drop(['col'], axis=1)