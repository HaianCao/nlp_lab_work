from typing import Dict
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from ..core.classification_interfaces import ModelInterface

class NeuralNetworkModel(ModelInterface):
    """
    Multi-layer Perceptron (Neural Network) model for text classification.
    """
    
    def __init__(self, hidden_layer_sizes=(100,), max_iter=500, random_state=42, **kwargs):
        """
        Initialize the Neural Network model.
        
        Args:
            hidden_layer_sizes: The ith element represents the number of neurons in the ith hidden layer.
            max_iter: Maximum number of iterations.
            random_state: Random state for reproducibility.
            **kwargs: Additional parameters for MLPClassifier.
        """
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            random_state=random_state,
            **kwargs
        )
        self.is_trained = False

    def fit(self, X, y):
        """
        Train the Neural Network model.
        
        Args:
            X: Training features
            y: Training labels
        """
        try:
            # Convert sparse matrix to dense if needed
            if hasattr(X, 'toarray'):
                X = X.toarray()
            
            self.model.fit(X, y)
            self.is_trained = True
        except Exception as e:
            print(f"Error training Neural Network model: {e}")
            raise

    def predict(self, X):
        """
        Predict using the trained Neural Network model.
        
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

    def get_training_info(self) -> Dict[str, int]:
        """
        Get training information from the neural network.
        
        Returns:
            Dictionary containing training iterations and convergence status
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet.")
        
        return {
            'n_iter': self.model.n_iter_,
            'n_layers': self.model.n_layers_,
            'n_outputs': self.model.n_outputs_,
            'converged': hasattr(self.model, 'loss_') and self.model.loss_ < 1e-4
        }