"""Model input layer L1: separate A/B candidate pools, shared Alpha158 definitions."""
from .candidates import (Candidate, RESPONSE_SCHEMA, baseline, build_catalog, build_prompt,
                         export_catalog, generate, parse_response)

__all__ = ['Candidate', 'RESPONSE_SCHEMA', 'baseline', 'build_catalog', 'build_prompt',
           'export_catalog', 'generate', 'parse_response']
