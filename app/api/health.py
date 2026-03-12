"""Health check endpoint."""

from flask_restx import Namespace

ns = Namespace("health", description="Service health check")
