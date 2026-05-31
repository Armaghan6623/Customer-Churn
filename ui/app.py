import os
from flask import Flask, jsonify


def create_app():
    app = Flask(__name__)

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"})

    @app.get("/api/forecast")
    def forecast_endpoint():
        # Placeholder endpoint. Your forecasting logic can be wired here later.
        return jsonify({"message": "forecast endpoint not wired yet"})

    return app


app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)

