# ### simple implementation of the mlflow class
import os
import pandas as pd
import yaml
from sklearn.ensemble import  *
from sklearn.linear_model import *
from utils.mlflow_class import *

# loading the config file
with open ("/home/vasanth/airflow/scripts/mlproject/config.yml", "r") as ymlfile:
            cfg = yaml.safe_load(ymlfile)
# loading the parameters required
exper_name = f"name='{cfg['experiment_details']['name']}'"
client = MlflowClient(tracking_uri = cfg['location']['tracking_uri'] , registry_uri = cfg['location']['registry_uri'])
model_name = cfg['model_parameters']['model_type']
print(type(model_name))

df = pd.read_csv(cfg['location']['yesterday_data'])


from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.3)

print('training values shape', train.shape)
print('training values ', train)
print('testing values shape', test.shape)
print( 'testing values are', test)
#preprocessing
#target = cfg['model_parameters']['target']
#features = df.columns
#features=features.drop(target)

#data = [df[features], df[target]]
