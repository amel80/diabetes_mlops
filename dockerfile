# Utiliser une image Python Slim pour une installation légère
FROM python:3.9-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier le fichier requirements.txt et installer les dépendances
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copier les scripts et le modèle
COPY scripts/app.py app.py
COPY models /app/models

# Exposer le port 8000 pour l'application FastAPI
EXPOSE 8000

# Lancer le serveur Uvicorn pour exécuter l'app FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

# FROM python:3.9-slim

# WORKDIR /app

# COPY requirements.txt requirements.txt
# RUN pip install -r requirements.txt

# COPY scripts/app.py app.py
# COPY models models

# EXPOSE 8000

# CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
