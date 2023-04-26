# NN-stock-demo

Demo project for stock prediction based on like-stocks using tensorflow.

Data fetched directly from Yahoo finance and stock tickers compliled into raw dataset.

Using pearson correlation the highest related stock is predicted on the unseen data.

3 keras layers were used with one dropout layer to build model.

Early stopping incorporated to reduce the length of training if accuracy does not improve.

Mean absolute error and Relative error on validation data used to compare the predicted results versus unseen data.
