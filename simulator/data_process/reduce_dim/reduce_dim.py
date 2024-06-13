from sklearn.decomposition import PCA
import pandas as pd
from sklearn.decomposition import KernelPCA
from sklearn.manifold import LocallyLinearEmbedding
import numpy as np



def pca(dim_num,train_embedding_list):
    train_embedding_df = pd.concat(train_embedding_list, ignore_index=True)
    numeric_columns = train_embedding_df.select_dtypes(include=[np.number]).columns
    sklearn_pca = PCA(n_components=dim_num)
    patient_embedding = sklearn_pca.fit_transform(train_embedding_df[numeric_columns].to_numpy())
    return patient_embedding

def kpca(dim_num,train_embedding_list, val_embedding_list, test_embedding_list):
    train_embedding_df = pd.concat(train_embedding_list, ignore_index=True)
    numeric_columns = train_embedding_df.select_dtypes(include=[np.number]).columns

    kernel_pca = KernelPCA(n_components=dim_num,
                           kernel='rbf')  # You can choose a different kernel, e.g., 'linear', 'poly', 'sigmoid', etc.

    patient_embedding = kernel_pca.fit_transform(train_embedding_df[numeric_columns].to_numpy())

    val_embedding_list = [kernel_pca.transform(val_df[numeric_columns].to_numpy()) for val_df in val_embedding_list]
    test_embedding_list = [kernel_pca.transform(test_df[numeric_columns].to_numpy()) for test_df in test_embedding_list]

    return patient_embedding, val_embedding_list, test_embedding_list


def lle(dim_num,train_embedding_list, val_embedding_list, test_embedding_list):
    train_embedding_df = pd.concat(train_embedding_list, ignore_index=True)
    numeric_columns = train_embedding_df.select_dtypes(include=[np.number]).columns

    lle_model = LocallyLinearEmbedding(n_components=dim_num,n_neighbors = 30)

    patient_embedding = lle_model.fit_transform(train_embedding_df[numeric_columns].to_numpy())

    # Apply LLE to other datasets
    val_embedding_list = [lle_model.transform(val_df[numeric_columns].to_numpy()) for val_df in val_embedding_list]
    test_embedding_list = [lle_model.transform(test_df[numeric_columns].to_numpy()) for test_df in test_embedding_list]
    return patient_embedding, val_embedding_list, test_embedding_list



if __name__ == '__main__':
    print()