from typing import Dict
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from ..core.classification_interfaces import ModelInterface

class GradientBoostingModel(ModelInterface):
    """
    Gradient Boosting Trees model for text classification.
    """
    
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42, **kwargs):
        """
        Initialize the Gradient Boosting Classifier model.
        
        Args:
            n_estimators: Number of boosting stages.
            learning_rate: Learning rate shrinks the contribution of each tree.
            max_depth: Maximum depth of the individual regression estimators.
            random_state: Random state for reproducibility.
            **kwargs: Additional parameters for GradientBoostingClassifier.
        """
        self.model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
            **kwargs
        )
        self.is_trained = False

    def fit(self, X, y):
        """
        Train the Gradient Boosting Classifier model.
        
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
            print(f"Error training Gradient Boosting model: {e}")
            raise

    def predict(self, X):
        """
        Predict using the trained Gradient Boosting Classifier model.
        
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

    def get_feature_importance(self, feature_names=None, top_n=10):
        """
        Get feature importance from the trained model.
        
        Args:
            feature_names: List of feature names
            top_n: Number of top features to return
            
        Returns:
            Dictionary or array of feature importance
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet.")
        
        try:
            importances = self.model.feature_importances_
            
            if feature_names:
                feature_importance = dict(zip(feature_names, importances))
                # Sort by importance
                sorted_features = sorted(feature_importance.items(), 
                                       key=lambda x: x[1], reverse=True)
                return dict(sorted_features[:top_n])
            else:
                # Return top indices and values
                indices = np.argsort(importances)[::-1][:top_n]
                return {
                    'indices': indices.tolist(),
                    'importances': importances[indices].tolist()
                }
        except Exception as e:
            print(f"Error getting feature importance: {e}")
            return {}

    def get_training_info(self) -> Dict[str, float]:
        """
        Get training information from the gradient boosting model.
        
        Returns:
            Dictionary containing training score and out-of-bag improvement
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet.")
        
        return {
            'train_score': self.model.train_score_[-1] if hasattr(self.model, 'train_score_') else 0.0,
            'n_estimators_used': len(self.model.estimators_),
            'oob_improvement': np.mean(self.model.oob_improvement_) if hasattr(self.model, 'oob_improvement_') else 0.0
        }