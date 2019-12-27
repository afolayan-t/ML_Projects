# -*- coding: utf-8 -*-
import numpy as np

def sigmoid(x):
    """Vectorized Sigmoid function"""
    x[x >= 50] = 50
    x[x <= -50] = -50
    return 1.0 / (1+np.exp(-x))

class mlp(object):
    """
    A two layer multilayer perceptron that implements regression. It has an input layer, 
        two hidden layers and a single output node. for each layer it has a set of weights 
        and bias upon which backprogation is run"""
    def __init__(self, input_size, hidden_size_1, hidden_size_2, std=1e-4, activation = 'relu'):
        """
        Initializes weights and biases for each layer of the network, taking in input_size(feature space),
        and first and second hidden layer sizes as inputs. It also takes in either 'relu' or 'sigmoid' as 
        string input to denote activation functions for each layer 
        """
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size_1)
        self.params['b1'] = np.zeros(hidden_size_1)
        self.params['W2'] = std * np.random.randn(hidden_size_1, hidden_size_2)
        self.params['b2'] = np.zeros(hidden_size_2)
        self.params['W3'] = std * np.random.randn(hidden_size_2, 1)
        self.params['b3'] = np.zeros(1)
        self.activation = activation
    
    def loss(self, X, y=None, reg = 0.0):
        """
        Input Shape (N, D)
        Runs the forward pass of the perceptron, and computes the loss. Runs backwards pass and computes 
        gradients for each layer. Returns loss and gradients.
        """
        
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        
        _, C = W2.shape
        _1, C1 = W3.shape
        N, D = X.shape
        
        # FORWARDS PASS
        
        #############################################################
        result = None
        
        z1 = np.dot(X, W1) + b1 #1st layer activation N*H1
        
        #1st layer nonlin N*H1
        if self.activation is 'relu':
            hidden_1 = np.maximum(0, z1)
        
        elif self.activation is 'sigmoid':
            hidden_1 = sigmoid(z1)
        else:
            raise ValueError('Unkown activation type')
        
        z2 = np.dot(hidden_1, W2) + b2 #2nd layer activation N*H2
        
        # 2nd layer non-lin H1*H2
        if self.activation is 'relu':
            hidden_2 = np.maximum(0, z2)
            
        elif self.activation is 'sigmoid':
            hidden_2 = sigmoid(z2)
            
        else:
            raise ValueError('Unknown activation type')
            
        result = np.dot(hidden_2, W3) + b3 #3rd layer activation, N*C1
        
        if y is None:
            return result
        
        # computing loss function mae
        #loss = np.mean(np.abs(y-result))
        loss = (np.mean((y-result), axis = 0)**2)**.5
        # adding regularization terms
        loss += .5 * reg * np.sum(W1 * W1)
        loss += .5 * reg * np.sum(W2 * W2)
        loss += .5 * reg * np.sum(W3 * W3)
        
        #############################################################
        
        # BACKWARDS PASS
        
        #############################################################
        grads = {}
        dloss = np.zeros((N,1))
        
        dloss = np.mean(2*(result-y))*(1/loss)
        dresult = np.tile(dloss, (N,1))
        
        dW3 = np.dot(hidden_2.T, dresult)/N
        #print(dW3.shape, "dW3", hidden_2.shape, "hidden_2")
        db3 = np.mean(dresult, axis = 0)
        
        #layer 2 gradient
        dhidden_2 = np.dot(dresult, W3.T)
        if self.activation is 'relu':
            dz2 = dhidden_2
            dz2[z2 <= 0] = 0
        
        elif self.activation is 'sigmoid':
            dz2 = (hidden_2*(1-hidden_2)) * dhidden_2
            
        else:
            raise ValueError('Unknown activation type')
            
        dW2 = np.dot(hidden_1.T, dz2)/N
        db2 = np.mean(dz2, axis = 0)
        
        #layer 1 gradient
        dhidden_1 = np.dot(dz2, W2.T)
        if self.activation is 'relu':
            dz1 = dhidden_1
            dz1[z1 <= 0] = 0
        
        elif self.activation is 'sigmoid':
            dz1 = (hidden_1*(1-hidden_1)) * dhidden_1
            
        else:
            raise ValueError('Unknown activation type')
            
        dW1 = np.dot(X.T, dz1)/N
        db1 = np.mean(dz1, axis = 0)
        
        grads['W3'] = dW3 + reg*W3
        grads['b3'] = db3
        grads['W2'] = dW2 + reg*W2
        grads['b2'] = db2
        grads['W1'] = dW1 + reg*W1
        grads['b1'] = db1
        
        #############################################################
        
        return loss, grads
    
    def train(self, X, y, X_val, y_val,
             learning_rate = 1e-3,learning_rate_decay= .95,
             reg=1e-5, num_epochs=10, batch_size = 200, verbose = False):
        """Runs training algorithmn, doing forward passes and implementing backprogration over some number of epochs."""
        
        num_train = X.shape[0]
        iterations_per_epoch = max(int(num_train/batch_size), 1)
        epoch_num = 0
        
        loss_history = []
        grad_magnitude_history = []
        train_acc_history = []
        val_acc_history = []
        
        np.random.seed(1)
        for epoch in range(num_epochs):
            
            perm = np.random.permutation(num_train)
            for i in range(iterations_per_epoch):
                
                X_batch = None
                y_batch = None
                
                idx = perm[i*batch_size:(i+1)*batch_size]
                X_batch = X[idx, :]
                y_batch = y[idx]
                
                loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
                loss_history.append(loss)
                
                for param in self.params:
                    self.params[param] -= grads[param] * learning_rate
                    
                grad_magnitude_history.append(np.linalg.norm(grads['W1']))
                
            train_acc =np.mean((self.predict(X_batch) == y_batch))
            val_acc = np.mean((self.predict(X_val) == y_val))
            train_acc_history.append(train_acc)
            val_acc_history.append(val_acc)
            
            if verbose:
                print('Epoch %d: loss %f, train_acc %f, val_acc %f'%(epoch+1, loss, train_acc, val_acc))
                
                
            learning_Rate = learning_rate*learning_rate_decay
            
        return {
                'loss_history': loss_history,
                'grad_magnitude_history': grad_magnitude_history,
                'train_acc_history': train_acc_history,
                'val_acc_history': val_acc_history
            }
        
    def predict(self, X):
        """Gives single value predictions on set of data examples input to the model"""
        y_pred = None
                    
            
        scores = self.loss(X)
        y_pred = scores
        return y_pred
