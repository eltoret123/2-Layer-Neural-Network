import numpy as np
import torch
from multiprocessing import Pool

data = torch.load("homework_features_256_50000.pth") #1

def sigma(z):
    return np.maximum(0,z)

W1 = np.random.randn(512, 256) 
b1 = np.zeros((1,512))
X = data['features'].numpy()
y = data['labels'].numpy().reshape(-1,1)
W2 = np.random.randn(1, 512)
b2 = np.zeros((1, 1))
lr_list = [1e-5, 1e-4, 1e-3]

def network(W1p,b1p,Xp,W2p,b2p): #2
    Z = Xp @ W1p.T + b1p
    s = sigma(Z)
    Y_hat = s @ W2p.T + b2p
    return Y_hat,s, Z

def MSE(ype,yhat): #3
    return np.average((ype - yhat)**2 )


def gradient(Xp, yp, yhatp, s, Z, W2p): #4
    N = Xp.shape[0]

    dL_dyhat = (2 / N) * (yhatp - yp)  


    dL_dW2 = dL_dyhat.T @ s  
    dL_db2 = np.sum(dL_dyhat, axis=0, keepdims=True)  


    dL_ds = dL_dyhat @ W2p 
    dL_dz = dL_ds * (Z > 0)  

    dL_dW1 = dL_dz.T @ Xp 
    dL_db1 = np.sum(dL_dz, axis=0, keepdims=True)  

    return dL_dW1, dL_db1, dL_dW2, dL_db2


def backtracking_line_search(W1, b1, W2, b2, dW1, db1, dW2, db2, X, y,
                             alpha=1e-4, beta=0.5):

    # Start with full step
    t = 1.0

    # Compute current loss
    y_hat, s, Z = network(W1, b1, X, W2, b2)
    loss = MSE(y, y_hat)

    # Directional derivative (gradient dot descent direction)
    grad_dot = (np.sum(dW1 * dW1) +
                np.sum(dW2 * dW2) +
                np.sum(db1 * db1) +
                np.sum(db2 * db2))

    # Backtracking loop
    while True:
        # Try new parameters
        W1_new = W1 - t * dW1
        b1_new = b1 - t * db1
        W2_new = W2 - t * dW2
        b2_new = b2 - t * db2

        y_hat_new, _, _ = network(W1_new, b1_new, X, W2_new, b2_new)
        loss_new = MSE(y, y_hat_new)

        # Armijo condition (slide formula)
        if loss_new <= loss - alpha * t * grad_dot:
            break

        # Shrink step size
        t *= beta

    return t






for epoch in range(1000): 
    Z = X @ W1.T + b1
    s = sigma(Z)
    y_hat = s @ W2.T + b2
    loss = MSE(y, y_hat)

    dW1, db1, dW2, db2 = gradient(X, y, y_hat, s, Z, W2)

    lr = backtracking_line_search(W1, b1, W2, b2, dW1, db1, dW2, db2, X, y)


    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2

    if epoch % 1 == 0:
        print(f"Epoch {epoch}: Loss = {loss:.4f}, W1 = {np.average(W1):.4f}, W2 = {np.average(W2):.4f}, b1 = {np.average(b1):.4f}, b2 = {np.average(b2):.4f}")


"""
I have tried optimised all constant W1, W2, b1, b2 and used a list of lr.

1: First we assign all the variables. We define the sigma function in the following way. 
From the given file we can find the Input and Output values and we will assign them to X and Y accordingly.
We will then assign varaible b1, b2, W1 and W2 to start with a random value from a N(0,1).
We also have a list of learning rate so that it can go through the list and choose the one that gives the lowes MSE.

2: Then we build the network function, this function is used to find the value of y_hat. 
The equation used is the equation given in the datasheet. 

3: Similar to 2, we will compute the MSE inputting the values we currently have to give us the output.

4: In this function we will calculate the gradient of each variable and how it affects the MSE.
We can calculate the gradient by taking the derivative of the MSE function. Therefore, we can calculate
calculate how each variable affects the MSE. After applying chain rule with respect to yhat. 
We can then multiply the differentated network relationship into the function and finish with a differential equation
that depends on each variable. The final result for each varaible can be seen in the code.

5: I have programmed the linear_search function so that a list of learning rates are taken in and it will try them all and output 
the ls which gives the smallest MSE.

6: Finally, we will use all the functions previously stated to optimise all the variables.
We can use a linear search to change the current value of each value constant and optimise towards a more precise value.
The equation I am using to optimise these values are W_new = W - lr * dL / dW. The value lr is a constant value that is found with the function 5.
The value dL is found from the gradient function. By iterating through this process method we optimise these constants.



"""