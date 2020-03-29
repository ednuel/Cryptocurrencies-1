# %%
import pandas as pd 

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.cluster import KMeans

import hvplot.pandas
import plotly.express as px 

# %% [markdown]
# # Data Preprocessing
# %%
# data load
crypto_df = pd.read_csv('./Resources/crypto_data.csv')
crypto_df.head()
# %%
crypto_df.dtypes
# %%
# Remove all cryptocurrencies that aren’t trading
active_crypto_df = crypto_df[crypto_df['IsTrading'] == True]
active_crypto_df.head(3)
# %%
# Remove all cryptocurrencies that don’t have an algorithm defined
active_crypto_df['Algorithm'].isnull().sum()

# %%
# Remove the IsTrading column
active_crypto_df = active_crypto_df.drop(columns = ['IsTrading'])

# %%
# Remove all cryptocurrencies with at least one null value

active_crypto_df.isnull().sum()
# %%
notnull_crypto_df = active_crypto_df.dropna()
notnull_crypto_df.head()
# %%
# Remove all cryptocurrencies without coins mined(TotalCoinsMined = 0)
cleaned_crypto_df = notnull_crypto_df[notnull_crypto_df['TotalCoinsMined'] != 0]
cleaned_crypto_df.head()

# %%
# Store the names of all cryptocurrencies on a DataFrame and use the original df index as the index for it
coins_name = pd.DataFrame(cleaned_crypto_df[['Unnamed: 0','CoinName']])
coins_name.set_index('Unnamed: 0', drop = True, inplace = True)
coins_name.head()
# %%
# Remove the CoinName column
cleaned_crypto_df = cleaned_crypto_df.drop(columns = ['CoinName'])
cleaned_crypto_df.head()
# %%
cleaned_crypto_df.dtypes

# %%
cleaned_crypto_df['TotalCoinSupply'] = cleaned_crypto_df['TotalCoinSupply'].astype('float')
# %%
# Create dummies variables for all of the text features, and store the resulting data on a DataFrame
X = pd.get_dummies(cleaned_crypto_df[['Algorithm','ProofType']])

# duummies extend to 98 features
# %%
# standardize all of the data from the X
scale_model = StandardScaler()
scaled_X = scale_model.fit_transform(X)   #ndarray

# %% [markdown]
# # PCA

# %%
# Reducing X DataFrame Dimensions Using PCA to 3 features
pca = PCA(n_components=3)
X_pca = pca.fit_transform(scaled_X)
print(f'The pca ratio is {pca.explained_variance_ratio_}')

# %%
pca.explained_variance_
# %%
pcs_df = pd.DataFrame(X_pca, index=cleaned_crypto_df['Unnamed: 0'], columns=['PC 1','PC 2','PC 3'])
pcs_df.head(10)

# %% [markdown]
# # Clustering by KMeans

# %%
# Create an elbow curve to find the best value for K, X-axis is K, y-axis is inertia
inertia_list = list()
k_value = list(range(1,16))

for k in k_value:
    k_model = KMeans(n_clusters=k, random_state=1)
    k_model.fit(pcs_df)
    inertia_list.append(k_model.inertia_)
# build a dataframe for plotting
elbow_df = pd.DataFrame({'K': k_value, 'Inertia': inertia_list})

# %%
# elbow curve
obj = elbow_df.hvplot.line(x = 'K', y = 'Inertia', xticks = k_value, title='Elbow Curve')
hvplot.show(obj)

# %% [markdown]
# Based on the elbow curve, at the point 5, the line shifts to a strong horizontal line.
# As a result, I chosed K=5 as the best estimate number of cluster in KMeans model.

# %%
# run the K-means algorithm to predict the K clusters for the cryptocurrencies’ data
model = KMeans(n_clusters=4, random_state=1)
predictions = model.fit_predict(pcs_df)


# %%
# combine all information with predicted cluster into a new DataFrame
clustered_df = cleaned_crypto_df.merge(pcs_df, on = 'Unnamed: 0')
clustered_df = clustered_df.merge(coins_name, on = 'Unnamed: 0')

clustered_df['Class'] = model.labels_

clustered_df.set_index('Unnamed: 0', drop = True, inplace = True)
clustered_df.head(10)
# %% [markdown]
# # Visualizing Results

# %%
# 3D scatter plot 
fig = px.scatter_3d(clustered_df, x= 'PC 1', y='PC 2',z='PC 3',
                    color='Class', symbol='Class', hover_name='CoinName',
                    hover_data=['Algorithm'])
fig.update_layout(legend = {'x':0,'y':1})
fig.show()

# %%
# create a hvplot table for all the current tradable cryptocurrencies
obj_table = clustered_df.hvplot.table(columns = ['CoinName', 'Algorithm', 
                                    'ProofType', 'TotalCoinSupply', 
                                    'TotalCoinsMined', 'Class'], width =500)

hvplot.show(obj_table)


# %%
# create a scatter plot to present the clustered data about cryptocurrencies 
obj = clustered_df.hvplot.scatter(x="TotalCoinsMined", y="TotalCoinSupply",
                                by = 'Class', hover_cols = ['CoinName'])

hvplot.show(obj)

# %%
