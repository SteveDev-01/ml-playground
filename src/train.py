import argparse
import yaml
import joblib
from data import load_data, split_data
from models import build_models
from sklearn.model_selection import GridSearchCV
from evaluate import evaluate_model

def main(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    df = load_data(config['data']['csv_path'])
    X_train, X_test, y_train, y_test = split_data(df, config['data']['target'])

    models = build_models()
    best_models = {}

    for name, model in models.items():
        grid = GridSearchCV(model, config['models'][name]['params'], cv=5)
        grid.fit(X_train, y_train)
        best_models[name] = grid.best_estimator_
        print(f"{name} best params: {grid.best_params_}")

    best_name = max(best_models, key=lambda n: evaluate_model(best_models[n], X_test, y_test)[config['metrics']['primary']])
    joblib.dump(best_models[best_name], config['paths']['model_output'])
    print(f"Best model: {best_name} saved to {config['paths']['model_output']}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config_path', required=True)
    args = parser.parse_args()
    main(args.config_path)
