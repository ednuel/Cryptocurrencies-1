# Cryptocurrencies

Use unsupervised Machine Learning techniques to analyze data on the cryptocurrencies traded on the market.

## Project Background

Present a report of what cryptocurrencies are on the trading market and how cryptocurrencies could be grouped toward creating a classification for developing a new investment product. Use unsupervised ML clustering algorithm to help determine cryptocurrencies groups.

## Project Objectives

- Prepare the data for dimensions reduction with PCA and clustering using K-means.

- Reduce data dimensions using PCA (Principal Component Anylysis) algorithms from sklearn.

- Predict clusters using cryptocurrencies data using the K-means algorithm form sklearn.

- Create some plots and data tables to present your results.

## Data Resources

- The cryptocurrencies data was retrieved from <https://min-api.cryptocompare.com/data/all/coinlist>

- The csv format data resource (/Resources/crypto_data.csv)

## Codes

- [Jupyter_Notebook](/crypto_PCA_kmeans.ipynb) 

- [Python](/crypto_PCA_kmeans.py)

### Results

- ![Elbow_curve](/Elbow_curve.PNG)

- ![2d_scatter_plot](/2d_scatter_plot.PNG)

- ![3d_scatter_plot](/3d_scatter_plot.PNG)

## Conclusion

Based on elbow curve figure, I decided to group preprocessed 533 cryptocurrencies into 4 clusters in oder to find their patterns. The 2d scatter figure shows clusters related to the number of available coins versus the total number of mined coins. The 3d scatter graph clearly displays different clusters distributed with 3 principal component variables.
