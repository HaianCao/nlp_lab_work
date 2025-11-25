from typing import List, Union, Dict

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from ..core.classification_interfaces import ModelInterface

class NaiveBayesModel(ModelInterface):
    """
    Gaussian Naive Bayes model for text classification.
    """
    
    def __init__(self):
        """Initialize the Gaussian Naive Bayes model."""
        self.model = GaussianNB()
        self.is_trained = False

    def fit(self, X, y):
        """
        Train the Gaussian Naive Bayes model.
        
        Args:
            X: Training features (dense array)
            y: Training labels
        """
        try:
            # Convert sparse matrix to dense if needed
            if hasattr(X, 'toarray'):
                X = X.toarray()
            
            self.model.fit(X, y)
            self.is_trained = True
        except Exception as e:
            print(f"Error training Gaussian Naive Bayes model: {e}")
            raise

    def predict(self, X):
        """
        Predict using the trained Gaussian Naive Bayes model.
        
        Args:
            X: Input features for prediction
            
        Returns:
            Predicted labels
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet.")
        
        try:
            # Convert sparse matrix to dense if needed
            if hasattr(X, 'toarray'):
                X = X.toarray()
                
            return self.model.predict(X)
        except Exception as e:
            print(f"Error during prediction: {e}")
            raise

    def evaluate(self, y_true, y_pred) -> Dict[str, float]:
        """
        Evaluate the model performance.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet.")
        
        try:
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall': recall_score(y_true, y_pred, zero_division=0),
                'f1_score': f1_score(y_true, y_pred, zero_division=0)
            }
            return metrics
        except Exception as e:
            print(f"Error during evaluation: {e}")
            raise
    
    def predict_proba(self, X):
        """
        Get prediction probabilities.
        
        Args:
            X: Input features
            
        Returns:
            Prediction probabilities for each class
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet.")
        
        try:
            # Convert sparse matrix to dense if needed
            if hasattr(X, 'toarray'):
                X = X.toarray()
                
            return self.model.predict_proba(X)
        except Exception as e:
            print(f"Error during probability prediction: {e}")
            raise