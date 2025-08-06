"""
Pytest configuration file
"""

import sys
import os
import pytest

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@pytest.fixture
def test_data_dir():
    """Return the path to the test data directory"""
    return os.path.join(os.path.dirname(__file__), 'data')
