class GoldenDatasetNotFoundError(FileNotFoundError):
    """Raise when the golden dataset is not found."""
    def __init__(self, path):
        super().__init__(f"Golden dataset not found: {path}")
