import pandas as pd
from model import HousePriceModel
import logging

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)


df_test = pd.read_csv('data/test.csv')
ids = df_test['Id']

model = HousePriceModel.getInstance()

predictions = model.predict(df_test)

submission_df = pd.DataFrame()
submission_df['Id'] = ids
submission_df['SalePrice'] = predictions

submission_df.to_csv('submission.csv', index=False)

logger.info('submission.csv written to disk!')
