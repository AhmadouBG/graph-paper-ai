"""Standalone subprocess worker for Marker PDF conversion.
Isolated in a subprocess because Marker (via sklearn/pyarrow) can crash
on certain Windows configurations with an access violation."""
import sys
from pathlib import Path

pdf_path = sys.argv[1]

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

artifact_dict = create_model_dict()
converter = PdfConverter(artifact_dict=artifact_dict)
rendered = converter(str(pdf_path))
text, _, images = text_from_rendered(rendered)
sys.stdout.write(text)
