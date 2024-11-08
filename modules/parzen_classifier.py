from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
import numpy as np

'''
Needs documentation!
'''

class ParzenClassifier(ClassifierMixin, BaseEstimator):
  def __init__(self, bandwidth=1.0, n_class=2, **params):
    super().__init__()
    self.n_class = n_class
    self.means = []
    self.x_class = []
    self.bandwidth = bandwidth 

  def fit(self, X: np.array, y=None):
    
    for c in range(self.n_class):
      _X = X[y==c]
      if _X.shape[0] > 0:
        self.means.append(_X.mean(0))
        self.x_class.append(_X)
    return self


  def windows_parzen(self, X, mean, h):
    parzen = np.zeros(X.shape)
    
    dif = np.abs(X - mean)
    parzen = dif

    summaries_parzen = np.sum(parzen, 1)/(mean.shape[0]*h)
    
    #kernel gaussian
    return np.exp(-.5*summaries_parzen**2)/((2*np.pi)**2)
  
  def prod_class(self, c, h):
      def prod(X, ):
        n, p = self.x_class[c].shape
        dif = np.abs(X - self.x_class[c])/h
        dif_gaussian = np.exp(-.5*(dif**2))/((2*np.pi)**(1/2))
        dif_parzen = np.prod(dif_gaussian, 1)

        return np.sum(dif_parzen)/(n*(h**p))
      return prod
  
  def prod_windows_parzen(self, X, h):
    summaries_parzen = np.array([np.apply_along_axis(self.prod_class(i, h), axis = 1, arr = X) for i in range(self.n_class)]).T
    
    return summaries_parzen
    # return np.exp(-.5*(summaries_parzen**2))/((2*np.pi)**(1/2))
  
  def prod_score_samples(self, X):
    return self.prod_windows_parzen(X, self.bandwidth)#.argmin(1)
  def predict(self, X):
    return self.prod_score_samples(X).argmax(1)
  def sigma(self, X):
    # mathematics great value
    s = np.std(X, 0)
    m_s = np.mean(s)
    best_h = 1.06*m_s/(X.shape[0]**(1/5))
    return best_h 