import numpy as np
import pandas as pd
import pymongo
from urllib.parse import quote_plus
from pymongo.mongo_client import MongoClient
from sklearn.metrics.pairwise import cosine_similarity
import nltk

nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
from gensim.models import Word2Vec
import logging


class Recommend:
    stop_words = set(stopwords.words("english"))
    stemmer = PorterStemmer()

    def __init__(self):
        logging.info("Model is loaded!")

    def generate_description(self, row):
        description = f"{row['color']} {row['glossiness']} tile"

        if row["floor"] == "Yes":
            description += " for floor"
        if row["wall"] == "Yes":
            description += " for wall"
        description += f" of length {row['length']}, width {row['width']} and thickness {row['thickness']}"
        description += f" price {row['price']}"
        return description

    ##################################################################################################################

    def preprocess_text(self, text):
        text = re.sub(r"\W", " ", str(text))
        text = re.sub(r"\s+", " ", text, flags=re.I)
        text = text.lower()
        text = [
            self.stemmer.stem(word)
            for word in text.split()
            if word not in self.stop_words
        ]
        return " ".join(text)

    ##################################################################################################################

    def connect_and_get_data(self, username, password, database_name, collection_name):
        uri = (
            "mongodb+srv://"
            + username
            + ":"
            + password
            + "@cluster0.dtmvzcl.mongodb.net/"
            + database_name
            + "?retryWrites=true&w=majority"
        )
        client = MongoClient(uri)
        try:
            client.admin.command("ping")
            print("Pinged your deployment. You successfully connected to MongoDB!")
        except Exception as e:
            print(e)

        db = client[database_name]
        collection = db[collection_name]
        documents = collection.find()
        data = list(documents)
        client.close()
        df = pd.DataFrame(data)
        df.drop(index=3, inplace=True)
        df_cleaned = df[
            [
                "length",
                "width",
                "thickness",
                "color",
                "glossiness",
                "floor",
                "wall",
                "price",
                "qty",
                "image",
            ]
        ]
        df_cleaned["description"] = df_cleaned.apply(self.generate_description, axis=1)
        df_cleaned["description"] = df_cleaned["description"].apply(
            self.preprocess_text
        )

        return df_cleaned

    #################################################################################################################

    def document_vector(self, words, model):
        word_vectors = model.wv
        valid_words = [word for word in words if word in word_vectors]

        if not valid_words:
            return np.zeros(model.vector_size)

        return np.mean([word_vectors[word] for word in valid_words], axis=0)

    #################################################################################################################

    def recommend_products(self, user_input, df):
        products = df.copy()
        corpus = [text.split() for text in products["description"]]
        model = Word2Vec(
            sentences=corpus, vector_size=100, window=5, min_count=1, workers=4
        )
        word_vectors = model.wv

        products["doc_vector"] = products["description"].apply(
            lambda x: self.document_vector(x.split(), model)
        )

        processed_user_input = self.preprocess_text(user_input)
        user_vector = self.document_vector(processed_user_input.split(), model)

        products["similarity"] = products["doc_vector"].apply(
            lambda x: cosine_similarity([user_vector], [x])[0][0]
        )

        recommended_products = products.sort_values(
            by="similarity", ascending=False
        ).iloc[:, :-3]

        recommended_products = recommended_products.values.tolist()
        return recommended_products

    ##############################################################################################################


def main():
    r = Recommend()
    username = quote_plus("ceramicoitems")
    password = quote_plus("zSiSvFpgZD56mpvE")
    database_name = "test"
    collection_name = "suppliers"
    recommended_tiles = r.recommend_products(
        "white and glossy",
        r.connect_and_get_data(username, password, database_name, collection_name),
    )
    logging.info(recommended_tiles)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()