# import the necessary libraries

import pandas as pd
import yaml


# loading the config file
with open ("/home/vasanth/airflow/scripts/mlproject/config.yml", "r") as ymlfile:
    cfg = yaml.safe_load(ymlfile)


# lodaing the today data from the raw folder
df = pd.read_csv(cfg['location']['raw_today_data'],index_col=False)


# loading the latest data with target
# df.to_csv(cfg['location']['today_data_with_target'], index = False)


# saving the todays data in the processed data location
# as it is testing, we removed only the target
target = cfg['model_parameters']['target']
features=df.columns
features=features.drop(target)
test = df[features]
test.to_csv(cfg['location']['today_data'],index = False)



# lodaing the yesterday data from the raw folder
df = pd.read_csv(cfg['location']['raw_yesterday_data'],index_col=False)


# saving the yesterday data with target to the processed data location
df.to_csv(cfg['location']['yesterday_data'], index = False)
