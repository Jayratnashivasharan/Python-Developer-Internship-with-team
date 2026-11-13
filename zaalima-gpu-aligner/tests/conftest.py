import pytest
from gpualign.simulate import write_dataset
from gpualign.index import build_index


@pytest.fixture(scope="session")
def dataset(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    info = write_dataset(str(d), genome_size=60_000, coverage=20)
    idx = str(d / "idx")
    build_index(info["fasta"], idx, verbose=False)
    info["index"] = idx
    return info
