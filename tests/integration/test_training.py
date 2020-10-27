import os


def test_run_training():
    import training_model
    assert os.path.exists('models')
    assert os.path.exists('models/stacked.model')
    assert os.path.exists('models/preprocessor.pipe')
