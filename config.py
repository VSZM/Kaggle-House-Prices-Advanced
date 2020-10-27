import os

list_of_environments = ['prod', 'test']

env = os.environ['ENV']

if env not in list_of_environments:
    raise ValueError(f'Unsupported Environment: |{env}|')


from config_common import *

if env == 'prod':
    from config_prod import *
elif env == 'test':
    from config_test import *

