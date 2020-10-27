import os

list_of_environments = ['prod', 'test']

env = os.environ['ENV']

if env not in list_of_environments:
    raise ValueError(f'Unsupported Environment: |{env}|')


import config_common

if env == 'prod':
    import config_prod.py
elif env == 'test':
    import config_test.py
