import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="session")
def tiny_model(tmp_path_factory):
    from grid.model.config import ModelConfig
    from grid.weights.make_tiny_model import make

    d = str(tmp_path_factory.mktemp("model"))
    make(d, ModelConfig(num_hidden_layers=3), seed=1)
    return d
