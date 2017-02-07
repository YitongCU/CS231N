import numpy as np
from random import shuffle

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
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  num_classes = W.shape[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = np.dot(X,W)
  scores = (scores.T - np.max(scores,1)).T
  for i in xrange(num_train):
    nominator = np.exp(scores[i,:])
    denominator = np.sum(np.exp(scores[i,:]))
    loss -= np.log(nominator[y[i]]/denominator)
    for j in xrange(num_classes):
      dW[:,j] += (nominator[j]/denominator)*X[i,:]
    dW[:,y[i]] -= X[i,:]

  loss /= num_train
  dW /= num_train
  loss += 0.5*reg*np.sum(W*W)
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
  num_train, num_dim = X.shape
  num_classes = W.shape[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = np.dot(X,W)
  # scores = scores.T - np.max(scores,1)
  # f = np.exp(scores.T) 
  # correct_scores = f[range(num_train),y] #1*N
  # col_sum = np.sum(f,1)
  # loss = np.sum(-np.log(correct_scores/col_sum))

  # mat = f.T/col_sum #
  # mat = mat.T
  # y_pred = np.zeros(mat.shape)
  # y_pred[range(num_train),y] = 1
  # dW = np.dot(X.T,mat-y_pred)

  # loss/=num_train
  # loss += 0.5*reg*np.sum(W*W)
  # dW /= num_train
  # dW += reg*W
  f = scores.T -  np.max(scores,1)
  f = f.T
  f_correct = scores[range(num_train),y]
  
  sum_col = np.log(np.sum(np.exp(scores),1)) # N*1
  
  loss = sum_col - f_correct # N*1
  loss = np.sum(loss)/num_train + 0.5*reg*np.sum(W*W)

  prob = np.exp(f).T / np.sum(np.exp(f),1)
  prob = prob.T
  y_pred = np.zeros(scores.shape)
  y_pred[range(num_train),y] = 1
  dW = X.T.dot(prob - y_pred)
  dW = dW/float(num_train) + reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

