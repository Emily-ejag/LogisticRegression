import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from copy import deepcopy

##############################
# GLOBALS
##############################
EPS = 0.0000005  # random epsilon

# Loads a data file from a provided file location.
def load_data(path):
    # Your code here:
    loaded_data = pd.read_csv(path)    
    return loaded_data


# Implements dataset preprocessing. For this assignment, you just need to implement normalization 
# of the three numerical features.

# save the mu and sigma values for normalizing test data later
saved_z_scale_values = {}

# probably want to use default argument training=True instead of checking inside, good idea!

#####################################################################
# PREPROCESSING DATA
#####################################################################

def preprocess_data(data, training):
    global saved_z_scale_values
    if isinstance(data, str):
        training = 'train' in data # set the flag depending if "train" in filepath
        data = load_data(data)  # load the file path
    else:
        raise ArgumentError('bad argument. expecting fpath')
        
    
    def z_score_standardization(col, series):
        #reuse saved_z_scale_values the values for validation data
        mu = series.mean()
        sigma = series.std()
        if training == True:
            saved_z_scale_values[col] = [mu,sigma]
        else:
            mu, sigma = saved_z_scale_values[col]
        return (series - mu) / sigma

    for col in data.columns:
        if col == 'Age' or col == 'Annual_Premium' or col == 'Vintage':
            data[col] = z_score_standardization(col, data[col])
    
    return data

#####################################################################
# END --------------PREPROCESSING DATA
#####################################################################

#####################################################################
# IMPORTANT FUNCTIONS
#####################################################################

# data - input dataframe with both x and y
# returns a numpy X matrix of Nxd (N = samples, d = variables) and y N-vector (each n in N price)
def split_df(data):
    # get the X matrix and y vector   
    X = data.drop('Response', axis=1)
    y = data.Response
    
    #X = pd.DataFrame(data.columns != ['Response'])
    #y = pd.DataFrame(data, columns = ['Response'])
    return X, y

# sigmoid function
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(z))

# runs a prediction to get the output on a set of weights 
def predict(X, w):
    return sigmoid(np.matmul(X, w))


def loss_fn(N, y_hat, y, w, lm, L2=True):
    loss_fn_sum = np.sum(-y * np.log(y_hat) - (1-y)*np.log(1 - y_hat))
    if L2:
        loss_fn_regl = np.sum(lm*w[1:]*w[1:])
    else:
        loss_fn_regl = np.sum(lm*np.abs(w[1:]))
    loss = ((1.0/N) * loss_fn_sum) + loss_fn_regl
    return loss


def lr_gd_train(X, y, lr, lm, L2=True):
    did_diverge = False
    losses = []
    last_loss = 1000000.0
    iterations = 0
    flag = True
    
    N, d = X.shape  # get samples and variables
    # column vec of weights
    w = np.zeros((d,), np.float32)
    # d = np.zeros((d, 1), np.float32)
    
    while flag:
        # Take a step in direction using gradient of the loss function
        # Step: w = w + alpha/N * [summation i-N (y_i - sigmoid(wTx_i)) * x_i]
        
        ##### vanilla gradient step #####
        # this method is the vanilla method (see below for numpy matrix method)
        # suma = np.zeros((d,), np.float32)
        # for i in range(N):
        #     wTx = np.dot(w.T, X[i])
        #     suma += ((float(y[i]) - sigmoid(wTx)) * X[i])
        # grad = (1.0/N)*suma
        
        ###### matrix gradient step #####
        ysXw = y - sigmoid(np.matmul(X, w))  # this should be now (N, 1)
        grad = (1.0/N) * np.matmul(X.T, ysXw) # X is (N, d) and we want (d, 1) so we do X.T*ysXw so (d, N) * (N, 1) = (d, 1)
        # print('diff', np.linalg.norm(grad - (1.0/N)*suma, 2))  # sanity check

        # Step: do gradient descent
        w -= lr*grad
        
        if L2: 
            # do L2 regularization
            # Step: w_j = w_j - alpha*lambda*w_j  
            w[1:] -= lr*lm*w[1:]  # exclude dummy variable
        else:
            # do L1 regularization
            # Step: w_j = sign(w_j) * max(|w_j| - alpha*lambda, 0)
            w[1:] = np.sign(w[1:])*np.maximum(np.abs(w[1:])-lr*lm, np.zeros(len(w[1:])))
            
        ##### vanilla version #####
        # for j in range(d):
        #     if j != 0:
        #         w[j] -= lr*lm*w[j]
        
        # vanilla loss function - missing regularization part
        # sumaloss = 0.0  # np.zeros((d,), np.float32)
        # for i in range (N):
        #     wTx2 = np.dot(w.T, X[i])
        #     # check for convergence: check if norm of gradient of loss function is below epsilon
        #     sumaloss += ((-float(y[i])*math.log(sigmoid(wTx2))-((1-float(y[i]))*math.log(1-sigmoid(wTx2)))))
        # loss = (1/N)*sumaloss
        # print(f'Loss {loss}')
        
        ##### matrix loss function #####
        y_hat = predict(X, w)  # np.matmul(X, w)
        loss = loss_fn(N, y_hat, y, w, lm, L2)
        # #        # print(f'Matrix loss {loss}')
        
        losses.append(float(loss)) 
        if (iterations >= 4000 and L2) or (iterations >= 12000 and not L2):  # different epochs for L1/L2
            print(f'Went over iterations. Exiting...')
            did_diverge = False
            flag = False
        elif abs(last_loss - loss) < EPS:
            print(f'Loss {loss} below {EPS}. Converged. Exiting...') 
            did_diverge = False
            flag = False
        
        iterations += 1
        last_loss = loss
    
    return w, losses, did_diverge


def accuracy(y, X, w):
    # count number of correct predictions between y and y_hat for each sample
    y_hat = np.round(predict(X, w))
    count_right = ((y == y_hat).sum())         
    acc = count_right/ y.shape[0]
    # print(acc)
    return acc


def plot_losses(losses, lambdas=None, name='plot.png'):
    plt.rcParams['figure.figsize'] = [15, 8]
    for i in range(len(losses)):
        lo = np.arange(0, len(losses[i]), 1)
        plt.plot(lo, losses[i], label=(None if learning_rates is None else f'i {lambdas[i]}'))
    # plt.ylim(0, 0.5)
    plt.xlabel('LOSSES')
    plt.ylabel('LOSS')
    plt.title('Convergence plot of various Lambdas at (LR=0.1)') 
    plt.legend()
    plt.savefig(name)
    # plt.show()
    plt.close()

    
# expecting all training accuracy, validation accuracy, and i values in separate arrays
def plot_acc(train_acc, val_acc, lambda_i, lr, name='accuracy.png'):
    plt.rcParams['figure.figsize'] = [15, 8]
    
    # plt.scatter(x=lambda_i, y=train_acc, label="Training Accuracy")
    # plt.scatter(x=lambda_i, y=val_acc, label="Validation Accuracy")
    
    width = 0.4
    plt.bar(x=np.array(lambda_i) - (width / 2), width=width, height=train_acc, label="Training Accuracy")
    plt.bar(x=np.array(lambda_i) + (width / 2), width=width, height=val_acc, label="Validation Accuracy")

    plt.xlabel('Value of i')
    plt.ylabel('Accuracy')
    plt.title(f'Training and Validation Accuracy for Different Regularization Parameters (LR={lr})') 
    plt.legend()
    plt.savefig(name)
    # plt.show()
    plt.close()
    
    
def plot_sparsity(num_weights, lambda_i, lr, name='sparsity.png'):
    plt.rcParams['figure.figsize'] = [15, 8]
    
    width = 0.4
    plt.bar(x=lambda_i, width=width, height=num_weights)

    plt.xlabel('Value of i')
    plt.ylabel('Number of Weights Approximately 0')
    plt.title(f'Number of Weights Approximately 0 for Different Regularization Parameters (LR={lr})') 
    # plt.legend()
    plt.savefig(name)
    # plt.show()
    plt.close()
    

def list_top_5(good_weights, X_cols, i, ival):
    # get the best weights for best lambda from 1a
    index = i.index(ival)  # pick lambda with i=ival
    best_val_weights = good_weights[index][1:]  # excluding dummy

    paired = list(zip(X_cols[1:], best_val_weights))  # excluding the dummy
    features = sorted(paired, key=lambda x: np.abs(x[1]), reverse=True)
    print('lambda', 10**ival, '. Top 5 Features:', features[:5])
    

def show_sparsity(good_weights, i, lr, name):
    zeros = []
    for ind, _ in enumerate(i):
        best_val_weights = good_weights[ind][1:]  # excluding dummy
        zero_vals = (best_val_weights <= 10e-6).sum()
        print('Zero Vals', zero_vals, 'out of', best_val_weights.shape)
        zeros.append(zero_vals)

    plot_sparsity(zeros, i, lr, name=name)
    
#####################################################################
# END --------------IMPORTANT FUNCTIONS
#####################################################################

#####################################################################
# L2 & L1 RUNNING FUNCTION
#####################################################################
    
def run_test(L2=True, noisy=False):
    print('##################')
    print(f'Running L2: {L2} and Noisy: {noisy} test') 
    # runs the specified test
    train_data = preprocess_data("IA2-train-noisy.csv" if noisy else "IA2-train.csv", True)
    val_data = preprocess_data("IA2-dev.csv", False)
    X, y = split_df(train_data)
    X_cols = X.columns
    X_val, y_val = split_df(val_data)

    # convert pandas DF to numpy mat for train
    assert X.columns[0] == 'dummy'
    X = X.to_numpy()
    y = y.to_numpy()

    # convert pandas DF to numpy mat for val
    assert X_val.columns[0] == 'dummy'
    X_val = X_val.to_numpy()
    y_val = y_val.to_numpy()


    training_losses = []
    training_acc = []
    val_acc = []
    lambdas = []
    good_weights = []

    # plot: i vs accuracy (#correct/total)
    # keep LR at 0.1,0.01 (we ran tests and saw this was pretty good for L1/L2 respectivly)
    if L2:
        lr = 0.1
    else:
        lr = 0.01
    
    i = [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2]
    lamList=[10**x for x in i]
    
    for l in lamList:
        #for m in range(5):
            #lr = np.power(10.0, -m)
        print(f'Testing {l}  with lr {lr}') 
        weights, losses, did_diverge = lr_gd_train(X, y, lr, l, True)

        # calculate final accuracy for training
        acc_train = accuracy(y, X, weights)
        print('Training acc', acc_train)
        training_acc.append(acc_train)

        # do validation accuracy for validation
        acc_val = accuracy(y_val, X_val, weights)
        val_acc.append(acc_val)

        # Add to plot data if curve did not diverge
        # and save the weights for validation data
        # if not did_diverge:
        training_losses.append(losses)
        # training_acc.append(acc)
        lambdas.append(l)
        good_weights.append(weights)
    
    # plot accuracy results
    suffix = '-' + ('L2' if L2 else 'L1') + '-' + ('noisy' if noisy else 'clean') + '.png'
    plot_acc(training_acc, val_acc, i, lr, 'acc' + suffix)
    print('Training acc', training_acc)
    print('Val acc', val_acc)
    
    # we selected different i for each test
    below = None
    at = None
    above = None
    if L2:
        below = -10
        at = -4
        above = 0
    else:
        below = -10
        at = -6
        above = -2
    
    # run the list of features
    print('At i', at, 'features')
    list_top_5(good_weights, X_cols, i, at)
    
    print('At i', below, 'features')
    list_top_5(good_weights, X_cols, i, below)
    
    print('At i', above, 'features')
    list_top_5(good_weights, X_cols, i, above)
    
    # plot sparsity results
    print('Plotting sparsity')
    show_sparsity(good_weights, i, lr, 'sparsity' + suffix)
    print(f'Finished L2: {L2} and Noisy: {noisy} test') 
    print('##################\n\n\n')
    
if __name__ == '__main__':
    print('Running tests!')
    # run L2 clean
    run_test(True, False)
    
    # run L2 noisy
    run_test(True, True)
    
    # run L1 clean
    run_test(False, False)
    
    print('All tests complete. Exiting...')
