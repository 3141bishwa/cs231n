import numpy as np
from random import shuffle

def softmax(X):
    
    if X.ndim > 1:
        diff = np.exp(X - X.max(axis=1)[:, np.newaxis])
        return diff/np.sum(diff, axis = 1)[:, np.newaxis]
        
    diff = np.exp(X - np.max(X))
    return diff/np.sum(diff)

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  num_train = X.shape[0]
  num_classes = W.shape[1]
    
  loss = 0.0
  dW = np.zeros_like(W)
  for i in xrange(num_train):
    # Compute vector of scores
    f_i = X[i].dot(W)
    out = softmax(f_i)
    
    diff = np.exp(out - np.max(out))
    
    sum_diff = np.sum(diff)
    
    score = diff[y[i]]/sum_diff
    
    loss += - np.log(score)
    
    
    # Compute gradient
    # Here we are computing the contribution to the inner sum for a given i.
    for k in range(num_classes):
      score_k = diff[y[k]]/sum_diff
      dW.T[i,k] += (score_k - (k == y[i])) * X[i]
    
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  dW /= num_train
  dW += reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

