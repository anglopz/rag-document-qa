"""Health check endpoint."""

import os

from flask import current_app
from flask_restx import Namespace, Resource, fields


def create_health_ns() -> Namespace:
    """Create the health namespace with all routes and models."""
    ns = Namespace("health", description="Service health check")

    health_model = ns.model(
        "Health",
        {
            "status": fields.String(description="Service status"),
            "models_loaded": fields.Boolean(description="Whether ONNX models are available"),
        },
    )

    @ns.route("")
    class HealthResource(Resource):
        """Service health check."""

        @ns.doc("health_check")
        @ns.marshal_with(health_model)
        def get(self):
            """Check service health and model availability."""
            embedding_path = current_app.config["EMBEDDING_MODEL_PATH"]
            qa_path = current_app.config["QA_MODEL_PATH"]

            models_loaded = (
                os.path.isfile(os.path.join(embedding_path, "model.onnx"))
                and os.path.isfile(os.path.join(qa_path, "model.onnx"))
            )

            return {
                "status": "healthy",
                "models_loaded": models_loaded,
            }

    return ns
