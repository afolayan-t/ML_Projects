# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 10:29:32 2019

@author: tolua
"""
import pandas as pd
import numpy as np


def add_column(X):
    """Adds a column of ones"""
    n_ = X.shape[0]
    return np.concatenate([X, np.ones((n_, 1))], axis = 1)
    
def predict(X, W):
    """Computes h(x, W)"""
    X_prime = add_column(X)
    return X_prime @ W

def loss(X, y, W):
    X_prime = add_column(X)
    loss = ((predict(X, W) - y)**2).mean()/2
    return loss

def loss_gradient(X, y, W):
    X_prime = add_column(X)
    loss_grad = ((predict(X, W) - y)*X_prime).mean(axis=0)[:, np.newaxis]
    return loss_grad

def gd(loss, loss_gradient, X, y, W_init, lr=0.01,n_iter = 1500):
    W_curr = W_init.copy()
    loss_values = []
    W_values = []
    
    for i in range(n_iter):
        loss_value = loss(X, y, W_curr)
        W_curr = W_curr - lr*loss_gradient(X, y, W_curr)
        
        loss_values.append(loss_value)
        W_values.append(W_curr)
        
    return W_curr, loss_values, W_values

def amen_count(series):
    amens = np.array([series]).T
    #print(amens.shape)
    num_amens = []
    for amen in amens:
        amen = amen[0].split(',')
        num_amens += [[len(amen)]]
    num_amens = np.array(num_amens)
    return num_amens


def main():
    
    data_df = pd.read_csv("data\\data.csv")
    
    test_ind = pd.read_csv("data\\test.csv")
    train_ind = pd.read_csv("data\\train.csv")
    val_ind = pd.read_csv("data\\val.csv")
    
    train_df = data_df.merge(train_ind, on = ["id"])
    val_df = data_df.merge(val_ind, on = ["id"])
    test_df = data_df.merge(test_ind, on=["id"])
    
    
    #train
    X_train = [train_df['bedrooms'].fillna(value = np.mean(train_df['bedrooms'],axis=0), axis = 0), train_df['bathrooms'].fillna(value = np.mean(train_df['bathrooms'],axis=0), axis = 0),train_df['beds'].fillna(value = np.mean(train_df['beds'],axis=0), axis = 0),train_df['accommodates'].fillna(value = np.mean(train_df['accommodates'],axis=0), axis = 0),train_df['review_scores_rating'].fillna(value = np.mean(train_df['review_scores_rating'],axis=0), axis = 0),train_df['review_scores_accuracy'].fillna(value = np.mean(train_df['review_scores_accuracy'],axis=0), axis = 0),train_df['review_scores_location'].fillna(value = np.mean(train_df['review_scores_location'],axis=0), axis = 0),train_df['review_scores_value'].fillna(value = np.mean(train_df['review_scores_value'],axis=0), axis = 0)]#train_df['square_feet'].fillna(value = 0.0, axis = 0)]
    y_train = np.array([train_df['price']]).T
    train_amen = amen_count(train_df['amenities'])
    #validation
    X_val = np.array([val_df['bedrooms'].fillna(value = np.mean(val_df['bedrooms'],axis=0), axis = 0), val_df['bathrooms'].fillna(value = np.mean(val_df['bathrooms'],axis=0), axis = 0), val_df['beds'].fillna(value = np.mean(val_df['beds'],axis=0), axis = 0), val_df['accommodates'].fillna(value = np.mean(val_df['accommodates'],axis=0), axis = 0),val_df['review_scores_rating'].fillna(value = np.mean(val_df['review_scores_rating'],axis=0), axis = 0),val_df['review_scores_accuracy'].fillna(value = np.mean(val_df['review_scores_accuracy'],axis=0), axis = 0),val_df['review_scores_location'].fillna(value = np.mean(val_df['review_scores_location'],axis=0), axis = 0),val_df['review_scores_value'].fillna(value = np.mean(val_df['review_scores_value'],axis=0), axis = 0)]).T
    y_val = np.array([val_df['price']]).T
    val_amen = amen_count(val_df['amenities'])
    #test
    test = np.array([test_df['bedrooms'].fillna(value = np.mean(test_df['bedrooms'],axis=0), axis = 0), test_df['bathrooms'].fillna(value = np.mean(test_df['bathrooms'],axis=0), axis = 0),test_df['beds'].fillna(value = np.mean(test_df['beds'],axis=0), axis = 0),test_df['accommodates'].fillna(value = np.mean(test_df['accommodates'],axis=0), axis = 0), test_df['review_scores_rating'].fillna(value = np.mean(test_df['review_scores_rating'],axis=0), axis = 0),test_df['review_scores_accuracy'].fillna(value = np.mean(test_df['review_scores_accuracy'],axis=0), axis = 0),test_df['review_scores_location'].fillna(value = np.mean(test_df['review_scores_location'],axis=0), axis = 0),test_df['review_scores_value'].fillna(value = np.mean(test_df['review_scores_value'],axis=0), axis = 0)]).T
    test_amen = amen_count(test_df['amenities'])

    y_val = np.array(y_val)
    X_train = np.array(X_train).T

    X_train = np.append(X_train, train_amen, axis = 1)
    X_val = np.append(X_val, val_amen, axis = 1)
    test = np.append(test, test_amen, axis = 1)

    print(X_train.shape, y_train.shape, X_val.shape, y_val.shape, test.shape)
    
    W_init = np.random.randn(10,1)
    
    print("Hang on, training my weights ...")
    
    result = gd(loss, loss_gradient, X_train, y_train, W_init, n_iter = 100000, lr = 1e-4)
    weights, loss_values, weight_values = result
    
    
    
    predictions = np.append(np.array(test_df['id'], dtype = 'int32').T.reshape((2051,1)), predict(test, weights), axis = 1)
    pred_df = pd.DataFrame(predictions, columns = ['id', 'price'])
    export_csv = pred_df.to_csv(r'tafol20_code\\pred.csv', index = None, header = True)
    
    print("Done.")
    
if __name__ == "__main__":
    main()