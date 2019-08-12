from flask import Flask, request, jsonify
import pandas as pd
import csv


def get_glove_vector(glove, words):
    return [glove.loc[w].values.tolist() for w in words]


def create_app(filename):
    app = Flask(__name__)
    glove = pd.read_table(filename, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)

    @app.route("/")
    def index():
        words = request.args.get("words", "")
        if words:
            words = words.split(",")
        else:
            words = []
        result = {
            "words": words,
            "vectors": get_glove_vector(glove, words),
        }
        return jsonify(result)

    return app


if __name__ == "__main__":
    print("loading glove vectors")
    glove_filename = "glove/glove.twitter.27B.25d.txt"
    app = create_app(glove_filename)
    app.run(debug=True, port=8989)
