from typing import List, Dict

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from ..core.classification_interfaces import ModelInterface

class TextClassifier(ModelInterface):
    """
    Text classifier using Logistic Regression as the underlying model.
    """
    
    def __init__(self, vectorizer):
        """
        Initialize the TextClassifier.
        
        Args:
            vectorizer: Text vectorization component (e.g., TF-IDF, Count vectorizer)
        """
        self.vectorizer = vectorizer
        self._model = None
        self.is_trained = False
    
    def fit(self, texts: List[str], labels: List[int]):
        """
        Train the classifier on texts and labels.
        
        Args:
            texts: List of input texts
            labels: List of corresponding labels
        """
        try:
            X = self.vectorizer.fit_transform(texts)
            self._model = LogisticRegression(solver='liblinear', random_state=42)
            self._model.fit(X, labels)
            self.is_trained = True
        except Exception as e:
            print(f"Error training TextClassifier: {e}")
            raise
    
    def predict(self, texts: List[str]) -> List[int]:
        """
        Make predictions on new texts.
        
        Args:
            texts: List of texts to classify
            
        Returns:
            List of predicted labels
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet. Call fit() first.")
        
        try:
            X = self.vectorizer.transform(texts)
            predictions = self._model.predict(X)
            return predictions.tolist()
        except Exception as e:
            print(f"Error during prediction: {e}")
            raise
    
    def evaluate(self, y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
        """
        Calculate evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary containing accuracy, precision, recall, and F1 score
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
    
    def get_feature_importance(self, feature_names: List[str], top_n: int = 10) -> Dict[str, float]:
        """
        Get feature importance from the trained model.
        
        Args:
            feature_names: List of feature names
            top_n: Number of top features to return
            
        Returns:
            Dictionary of top features and their coefficients
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet.")
        
        try:
            coefficients = self._model.coef_[0]
            feature_importance = dict(zip(feature_names, coefficients))
            
            # Sort by absolute value of coefficients
            sorted_features = sorted(feature_importance.items(), 
                                   key=lambda x: abs(x[1]), reverse=True)
            
            return dict(sorted_features[:top_n])
        except Exception as e:
            print(f"Error getting feature importance: {e}")
            return {}

    def predict_proba(self, texts: List[str]) -> List[List[float]]:
        """
        Get prediction probabilities.
        
        Args:
            texts: List of texts to classify
            
        Returns:
            List of probability arrays for each class
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet.")
        
        try:
            X = self.vectorizer.transform(texts)
            probabilities = self._model.predict_proba(X)
            return probabilities.tolist()
        except Exception as e:
            print(f"Error during probability prediction: {e}")
            raise