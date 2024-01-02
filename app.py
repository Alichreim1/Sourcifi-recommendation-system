from urllib.parse import quote_plus
from flask import Flask, render_template, request
from model.model import Recommend

app = Flask(__name__)
recommendation_model = Recommend()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/recommend", methods=["POST"])
def recommend():
    user_input = request.form["user_input"]
    username = quote_plus("ceramicoitems")
    password = quote_plus("zSiSvFpgZD56mpvE")
    database_name = "test"
    collection_name = "suppliers"

    data = recommendation_model.connect_and_get_data(
        username, password, database_name, collection_name
    )

    recommended_tiles = recommendation_model.recommend_products(user_input, data)
    recommended_tiles = recommended_tiles[:25]

    return render_template("index.html", recommended_tiles=recommended_tiles)


if __name__ == "__main__":
    app.run(debug=True)
