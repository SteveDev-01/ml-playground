from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def build_models():
    return {
        'rf': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier())
        ]),
        'lr': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression())
        ])
    }
