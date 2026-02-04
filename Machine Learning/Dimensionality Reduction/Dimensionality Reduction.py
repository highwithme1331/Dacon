#PCA
U, S, Vt = np.linalg.svd(scaled_train, full_matrices=False)

n_components = 3
components_train = Vt[:n_components]

pca_train = np.dot(scaled_train, components_train.T)
pca_test = np.dot(scaled_test, components_train.T)



#PCA(sklearn)
from sklearn.decomposition import PCA

pca = PCA(n_components=3)
pca.fit(scaled_train)

train_pca = pca.transform(scaled_train)
test_pca = pca.transform(scaled_test)



#LDA + elbow
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

lda = LDA()

lda.fit(scaled_train, y_train)
lda_transformed_train = lda.transform(scaled_train)

lda_columns = [f'lda{i+1}' for i in range(lda_transformed_train.shape[1])]
lda_df = pd.DataFrame(lda_transformed_train, columns=lda_columns)

plt.figure(figsize=(10, 6))
plt.plot(range(1, lda_transformed_train.shape[1]+1), lda.explained_variance_ratio_)
plt.show()



#PCA elbow
pca = PCA()

pca.fit(scaled_train)
pca_transformed_train = pca.transform(scaled_train)
    
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(pca.explained_variance_ratio_)+1), pca.explained_variance_ratio_)
plt.show()