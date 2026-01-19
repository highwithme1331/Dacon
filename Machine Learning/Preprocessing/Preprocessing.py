#결측치 탐지
train.isnull().sum()


import seaborn as sns
import matplotlib.pyplot as plt

ax = sns.boxplot(train[col])
plt.show()



#중복 데이터 식별
drop_index_df = train.drop('index', axis=1)
duplicated_df = drop_index_df[drop_index_df.duplicated()]

duplicated_indices = duplicated_df.index
final_train = train.drop(duplicated_indices, axis=0)