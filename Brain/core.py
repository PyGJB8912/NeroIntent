from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import joblib
import os

class Model:
    def __init__(self, model_path="model.joblib"):
        self.model_path = model_path
        self.model = None
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)

    def train(self, data):
        x = []
        y = []
        for intent, phrases in data.items():
            for phrase in phrases:
                x.append(phrase)
                y.append(intent)

        # Création du pipeline de classification
        self.model = make_pipeline(
            TfidfVectorizer(),
            LogisticRegression()
        )

        # Entraînement
        self.model.fit(x, y)

        # Sauvegarde du modèle
        joblib.dump(self.model, self.model_path)

    def detect_intention(self, text):
        if self.model is None:
            raise ValueError("Le modèle n'a pas été entraîné ou chargé.")
        return self.model.predict([text])[0]
