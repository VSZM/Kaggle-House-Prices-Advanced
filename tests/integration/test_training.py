import os
import config

def test_run_training():
    import training_model
    assert os.path.exists('models')
    assert os.path.exists('models/' + config.model_to_use)
    assert os.path.exists('models/' + config.pipe)
