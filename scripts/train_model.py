import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score
import pickle
import mlflow
import mlflow.sklearn
from datetime import datetime
from sklearn.model_selection import train_test_split


print(mlflow.get_tracking_uri())
# Définir l'URL de suivi MLflow
mlflow.set_tracking_uri("https://opulent-space-tribble-v6rv7qrp9g9vcp994-5000.app.github.dev/#/experiments/0?searchFilter=&orderByKey=attributes.start_time&orderByAsc=false&startTime=ALL&lifecycleFilter=Active&modelVersionFilter=All+Runs&datasetsFilter=W10%3D")

# Générer un nom d'expérimentation dynamique basé sur la date et l'heure
experiment_name = f"experiment_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

# Définir l'expérimentation dynamique
mlflow.set_experiment(experiment_name)

# Charger les données

reference_data = pd.read_csv('data/reference_data.csv')
new_data = pd.read_csv('data/new_data.csv')

# Combiner les données
df= pd.concat([reference_data, new_data], ignore_index=True)

# Séparer les caractéristiques et la cible

X = df.drop('Outcome', axis=1)
y = df['Outcome']
# Diviser les données en ensembles d'entraînement et de test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Prétraitement des caractéristiques numériques

numeric_features = X.columns
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Transformer les colonnes

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features)
    ])
# Créer le pipeline

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])
# Démarrer une exécution MLflow
with mlflow.start_run() as run:
    run_id = run.info.run_id
    print("Run ID:", run_id)
    
    # Log des paramètres
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_param("data_source", "reference_data + new_data")
    
    # Entraînement du modèle
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    # Calcul et log des métriques
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("f1_score", f1)
    
    # Enregistrement du modèle
    mlflow.sklearn.log_model(pipeline, "model")
    print("Modèle entraîné et logué dans MLflow.")

# Sauvegarde du modèle localement avec Pickle
with open('models/pipeline.pkl', 'wb') as f:
    pickle.dump(pipeline, f)
print("Modèle sauvegardé localement sous 'models/pipeline.pkl'.")

# Sélectionner le meilleur modèle basé sur le F1-score
experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
runs = mlflow.search_runs(experiment_ids=[experiment_id])

best_run = runs.loc[runs['metrics.f1_score'].idxmax()]
best_model_uri = f"runs:/{best_run.run_id}/model"
best_model = mlflow.sklearn.load_model(best_model_uri)

# Prédictions avec le meilleur modèle
y_pred_best_model = best_model.predict(X_test)
print(f"Meilleur modèle (Run ID: {best_run.run_id}) chargé pour les prédictions futures.")

# Résultat final
print("Pipeline MLflow terminé avec succès.")