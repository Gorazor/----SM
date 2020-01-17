import numpy as np
from sklearn.base import BaseEstimator
from sklearn.preprocessing import normalize as normalize_
class KMSM(BaseEstimator):
    """Subspace Method (SM)
    Classification method using Subspace.
    Parameters
    ----------
    n_dimension : int
        Number of dimension of subspace.
    Attributes
    ----------
    _dict :list, each item is a 2D array size of (n_samples, n_features)
        Subspace of N classes.
    """
    def __init__(self, kargs):
        for parameter, value in kargs.items():
            setattr(self,parameter, value)

    def euclidean_grammian(self,X,Y):
        return (X**2).sum(axis=1)[:,np.newaxis] + (Y**2).sum(axis=1)[np.newaxis,:] - 2 * X.dot(Y.T)
    
    def fit(self, X, y):
        """Fit the model with X.
        Parameters
        ----------
        X : list, each item is  size of (n_samples, n_features)
        y : label, each item  
            
        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self.n_dict = len(X)
        self.label = np.array(y)
        if self.is_normalize:
            self.dict_X = [normalize_(i) for i in X] 
        else:
            self.dict_X = [np.array(i,dtype='float')for i in X]
              

        tmpt_K = []
        self.sigma = 0.0
        for i in range(self.n_dict):
            tmpt_K.append(self.euclidean_grammian(self.dict_X[i],self.dict_X[i]))
            self.sigma += 1./np.mean(tmpt_K[i])
        self.sigma /= self.n_dict
        self.sigma *= self.scale
        self.tmpt_K = tmpt_K
        self.dict_K = [np.exp(-self.sigma * i) for i in tmpt_K]
        
        self.dict_U =[]    
        for i in range(self.n_dict):
            _,s,v = np.linalg.svd(self.dict_K[i],False)
            self.dict_U.append(v[:self.n_dimension_dict,:]/np.sqrt(s[:self.n_dimension_dict,np.newaxis]))
        return self

    def predict(self, X):
        """
        Perform classification on samples in X.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        Returns
        -------
        y_pred : array, shape (n_samples,)
            Class labels for samples in X.
        """
        self.n_query = len(X)
        if self.is_normalize:
            self.query_X = [normalize_(i) for i in X]
        else:
            self.query_X = [np.array(i,dtype='float')for i in X]

        self.query_K = []
        self.query_V = []
        
        for i in range(self.n_query):
            #self.query_K.append(euclidean_grammian(X[i],X[i]))
            self.query_K.append(np.exp(-self.sigma *self.euclidean_grammian(self.query_X[i],self.query_X[i])))
            _,s,v = np.linalg.svd(self.query_K[i],False)
            self.query_V.append(v[:self.n_dimension_query,:]/np.sqrt(s[:self.n_dimension_query,np.newaxis]))

        c_mat = np.zeros([self.n_query,self.n_dict])
        for i in range(self.n_query):
            for j in range(self.n_dict):
                K = np.exp(-self.sigma *self.euclidean_grammian(self.query_X[i],self.dict_X[j]))
                _, s, _ = np.linalg.svd(self.query_V[i]@K@self.dict_U[j].T)
                c_mat[i,j] = s.max()
        y_pred = self.label[np.argmax(c_mat,axis=1)]
        return y_pred
    

    def score(self, X, y):
        return np.mean(self.predict(X) == y)

    def get_params(self, deep=True):
        return {'n_dimension_dict': self.n_dimension_dict,'n_dimension_query': self.n_dimension_query}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self,parameter, value)
        return self
    
