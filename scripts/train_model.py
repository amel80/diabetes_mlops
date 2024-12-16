import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle
import mlflow
import mlflow.sklearn

# Charger les données

reference_data = pd.read_csv('data/reference_data.csv')
new_data = pd.read_csv('data/new_data.csv')
# Combiner les données

df= pd.concat([reference_data, new_data], ignore_index=True)

# Séparer les caractéristiques et la cible

X = df.drop('Outcome', axis=1)
y = df['Outcome']
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
# Démarrer une nouvelle expérimentation MLflow

mlflow.start_run()
run_id = mlflow.active_run().info.run_id
print("Run ID:", run_id)

# Enregistrer les paramètres, la métrique et le modèle avec MLflow
mlflow.log_param("model_type", "RandomForest")
mlflow.log_param("data_source", "reference_data + new_data")

# Entraîner le modèle

pipeline.fit(X, y)

# Enregistrer le modèle avec MLflow
mlflow.sklearn.log_model(pipeline, "model")

# Sauvegarder le modèle localement avec pickle (facultatif si vous voulez aussi un fichier pickle)

with open('models/pipeline.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

# Terminer l'exécution de MLflow
mlflow.end_run()

print("Model training and logging completed.")