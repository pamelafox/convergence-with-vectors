def pytest_addoption(parser):
    parser.addoption(
        "--real-models",
        action="store_true",
        help="Run the browser tests against the real embedding models and data/convergence.db instead of a fake embedder.",
    )
