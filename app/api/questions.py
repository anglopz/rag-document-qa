"""Question answering endpoint."""

from flask_restx import Namespace

ns = Namespace("questions", description="Ask questions about uploaded documents")
