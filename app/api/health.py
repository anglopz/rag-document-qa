"""Health check endpoint."""

from flask import current_app
from flask_restx import Namespace, Resource, fields

from app.services.embedding_service import _find_onnx_model


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
            try:
                _find_onnx_model(current_app.config["EMBEDDING_MODEL_PATH"])
                _find_onnx_model(current_app.config["QA_MODEL_PATH"])
                models_loaded = True
            except FileNotFoundError:
                models_loaded = False

            return {
                "status": "healthy",
                "models_loaded": models_loaded,
            }

    return ns
