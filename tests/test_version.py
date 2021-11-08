import cascade_fer
import logging

logging.basicConfig(level=logging.INFO)


def test_version():
    logging.info(f"Version: {cascade_fer.__version__}")
    assert cascade_fer.__version__ == "0.1.2"
