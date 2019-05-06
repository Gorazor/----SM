import numpy as np
from sklearn.base import BaseEstimator
from sklearn.preprocessing import normalize as normalize_
class MSM(BaseEstimator):
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

        
#     def __init__(self, n_dim_dict,n_dim_query=None,is_normalize=False):
#         self.n_dimension_dict = n_dim_dict
#         if n_dim_query is None:
#             self.n_dimension_query = n_dim_dict 
#         else:
#             self.n_dimension_query = n_dim_query
#         self.is_normalize = is_normalize
    
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
        self.dict = []
        self.label = np.array(y)
        if self.is_normalize:
            X = [normalize_(i) for i in X]    
        for i in range(len(X)):
            _,_,v = np.linalg.svd(X[i],False)
            self.dict.append(v[:self.n_dimension_dict,:])
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
        self.query=[]
        if self.is_normalize:
            X = [normalize_(i) for i in X]

        for x in X:
            _,_,v = np.linalg.svd(x,False)
            self.query.append(v[:self.n_dimension_query,:])
        c_mat = np.zeros([len(self.query),len(self.dict)])
        for i in range(len(self.query)):
            for j in range(len(self.dict)):
                _, s, _ = np.linalg.svd(self.query[i]@self.dict[j].T)
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