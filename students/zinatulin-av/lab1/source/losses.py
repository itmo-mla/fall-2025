import numpy as np


class Loss():
  def __init__(self):
    pass

  def loss():
    raise NotImplementedError("Please Implement this method")

  def derivative():
    raise NotImplementedError("Please Implement this method")


class LogLoss(Loss):
  def __init__(self):
    pass

  def loss(self, x):
    return np.log(1 + np.exp(-x))

  def derivative(self, x):
    return -1 / (1 + np.exp(x))