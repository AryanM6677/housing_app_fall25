import sys
sys.path.insert(0, '.')

import pandas as pd
import pickle
import os
from sklearn.decomposition import PCA
# from sklearn.model_selection import clone
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score
import optuna
from optuna.pruners import MedianPruner
import warnings
warnings.filterwarnings('ignore')

from src.data.load_data import get_train_test_data

print("=" * 80)
print("HEART FAILURE PREDICTION - 16 EXPERIMENTS")
print("=" * 80)

os.makedirs('models/saved_models', exist_ok=True)
os.makedirs('experiments', exist_ok=True)

# Load data
print("\nLoading data...")
X_train, X_test, y_train, y_test, n_features = get_train_test_data()
print(f"Data ready: {X_train.shape[0]} training, {X_test.shape[0]} testing")

# Models
models = {
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
    'RandomForest': RandomForestClassifier(random_state=42, n_jobs=-1),
    'GradientBoosting': GradientBoostingClassifier(random_state=42),
    'SVM': SVC(probability=True, random_state=42)
}

all_results = []
exp_counter = 0

# Train each model 4 ways
for model_name, base_model in models.items():
    print(f"\n{model_name}")
    
    # Exp 1: PCA + No tuning
    exp_counter += 1
    pca = PCA(n_components=10)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    model = clone(base_model)
    model.fit(X_train_pca, y_train)
    y_pred = model.predict(X_test_pca)
    f1 = f1_score(y_test, y_pred)
    
    all_results.append({
        'experiment_id': exp_counter,
        'model': model_name,
        'pca_applied': True,
        'hyperparameter_tuning': False,
        'f1_score': f1,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred)
    })
    
    with open(f'models/saved_models/exp_{exp_counter:02d}_{model_name}.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print(f"  [Exp {exp_counter}] PCA+No Tune: F1={f1:.4f}")
    
    # Exp 2: PCA + Tuning
    exp_counter += 1
    
    def objective(trial):
        if model_name == 'LogisticRegression':
            C = trial.suggest_float('C', 0.001, 100, log=True)
            m = LogisticRegression(C=C, max_iter=1000, random_state=42)
        elif model_name == 'RandomForest':
            n_est = trial.suggest_int('n_estimators', 50, 150)
            max_d = trial.suggest_int('max_depth', 5, 15)
            m = RandomForestClassifier(n_estimators=n_est, max_depth=max_d, random_state=42, n_jobs=-1)
        elif model_name == 'GradientBoosting':
            n_est = trial.suggest_int('n_estimators', 50, 150)
            lr = trial.suggest_float('learning_rate', 0.01, 0.3)
            m = GradientBoostingClassifier(n_estimators=n_est, learning_rate=lr, random_state=42)
        else:
            C = trial.suggest_float('C', 0.1, 100, log=True)
            kernel = trial.suggest_categorical('kernel', ['linear', 'rbf'])
            m = SVC(C=C, kernel=kernel, probability=True, random_state=42)
        
        m.fit(X_train_pca, y_train)
        pred = m.predict(X_test_pca)
        return f1_score(y_test, pred)
    
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction='maximize', sampler=sampler, pruner=MedianPruner())
    study.optimize(objective, n_trials=8, show_progress_bar=False)
    
    best_params = study.best_params
    best_f1 = study.best_value
    
    if model_name == 'LogisticRegression':
        model = LogisticRegression(C=best_params['C'], max_iter=1000, random_state=42)
    elif model_name == 'RandomForest':
        model = RandomForestClassifier(n_estimators=best_params['n_estimators'], max_depth=best_params['max_depth'], random_state=42, n_jobs=-1)
    elif model_name == 'GradientBoosting':
        model = GradientBoostingClassifier(n_estimators=best_params['n_estimators'], learning_rate=best_params['learning_rate'], random_state=42)
    else:
        model = SVC(C=best_params['C'], kernel=best_params['kernel'], probability=True, random_state=42)
    
    model.fit(X_train_pca, y_train)
    y_pred = model.predict(X_test_pca)
    
    all_results.append({
        'experiment_id': exp_counter,
        'model': model_name,
        'pca_applied': True,
        'hyperparameter_tuning': True,
        'f1_score': best_f1,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred)
    })
    
    with open(f'models/saved_models/exp_{exp_counter:02d}_{model_name}.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print(f"  [Exp {exp_counter}] PCA+Tune: F1={best_f1:.4f}")
    
    # Exp 3: No PCA + No tuning
    exp_counter += 1
    model = clone(base_model)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    
    all_results.append({
        'experiment_id': exp_counter,
        'model': model_name,
        'pca_applied': False,
        'hyperparameter_tuning': False,
        'f1_score': f1,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred)
    })
    
    with open(f'models/saved_models/exp_{exp_counter:02d}_{model_name}.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print(f"  [Exp {exp_counter}] NoPCA+No Tune: F1={f1:.4f}")
    
    # Exp 4: No PCA + Tuning
    exp_counter += 1
    
    def objective(trial):
        if model_name == 'LogisticRegression':
            C = trial.suggest_float('C', 0.001, 100, log=True)
            m = LogisticRegression(C=C, max_iter=1000, random_state=42)
        elif model_name == 'RandomForest':
            n_est = trial.suggest_int('n_estimators', 50, 150)
            max_d = trial.suggest_int('max_depth', 5, 15)
            m = RandomForestClassifier(n_estimators=n_est, max_depth=max_d, random_state=42, n_jobs=-1)
        elif model_name == 'GradientBoosting':
            n_est = trial.suggest_int('n_estimators', 50, 150)
            lr = trial.suggest_float('learning_rate', 0.01, 0.3)
            m = GradientBoostingClassifier(n_estimators=n_est, learning_rate=lr, random_state=42)
        else:
            C = trial.suggest_float('C', 0.1, 100, log=True)
            kernel = trial.suggest_categorical('kernel', ['linear', 'rbf'])
            m = SVC(C=C, kernel=kernel, probability=True, random_state=42)
        
        m.fit(X_train, y_train)
        pred = m.predict(X_test)
        return f1_score(y_test, pred)
    
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction='maximize', sampler=sampler, pruner=MedianPruner())
    study.optimize(objective, n_trials=8, show_progress_bar=False)
    
    best_params = study.best_params
    best_f1 = study.best_value
    
    if model_name == 'LogisticRegression':
        model = LogisticRegression(C=best_params['C'], max_iter=1000, random_state=42)
    elif model_name == 'RandomForest':
        model = RandomForestClassifier(n_estimators=best_params['n_estimators'], max_depth=best_params['max_depth'], random_state=42, n_jobs=-1)
    elif model_name == 'GradientBoosting':
        model = GradientBoostingClassifier(n_estimators=best_params['n_estimators'], learning_rate=best_params['learning_rate'], random_state=42)
    else:
        model = SVC(C=best_params['C'], kernel=best_params['kernel'], probability=True, random_state=42)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    all_results.append({
        'experiment_id': exp_counter,
        'model': model_name,
        'pca_applied': False,
        'hyperparameter_tuning': True,
        'f1_score': best_f1,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred)
    })
    
    with open(f'models/saved_models/exp_{exp_counter:02d}_{model_name}.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print(f"  [Exp {exp_counter}] NoPCA+Tune: F1={best_f1:.4f}")

# Save results
print(f"\n{'='*80}")
print("ALL 16 EXPERIMENTS COMPLETED!")
print(f"{'='*80}\n")

results_df = pd.DataFrame(all_results)
print(results_df.to_string(index=False))
print(f"\nMean F1: {results_df['f1_score'].mean():.4f}")
print(f"Best F1: {results_df['f1_score'].max():.4f}")

results_df.to_csv('experiments/16_experiments_results.csv', index=False)
print(f"\nSaved to: experiments/16_experiments_results.csv")
