# importing the necessary libraries
import os
import mlflow
from itertools import product
from utils.evaluation import *
from mlflow.tracking import MlflowClient
from utils.hyperopt_model_selection import *
import yaml

with open ("/home/vasanth/airflow/scripts/mlproject/config.yml", "r") as ymlfile:
    cfg = yaml.safe_load(ymlfile)

# creating the mlflow client
client = MlflowClient(tracking_uri = cfg['location']['tracking_uri'] , registry_uri = cfg['location']['registry_uri'])
os.chdir(cfg['location']['mlflow_artifacts'])


print('getting cuurrent location', os.getcwd())

class MlflowExperiment:
    ''' it is class for the mlflow experiment tracking '''
    def __init__(self,data_location, data_version, train_data,test_data,model_name, model_parameters, experiment_name, problem_type):
        # intializing the parameters required
        self.train_data = train_data
        self.model_params = model_parameters
        self.model = model_name
        self.experiment_name = experiment_name
        self.test_data = test_data
        self.problem_type = problem_type
        self.data_location = data_location
        self.data_version = data_version
        # creating the parameters combinations
        parameters, values = zip(*model_parameters.items())
        self.result = [dict(zip(parameters, value)) for value in product(*values)]

    def mlflow_runs(self):
        ''' it will create runs and store the model parameters and the metrics in the mlflow'''
        # setting the sqlite tracking uri
        # for postgres database : mlflow.set_tracking_uri('postgresql://postgres:postgres@localhost/mlflow')
        mlflow.set_tracking_uri('sqlite:///mlflow.db')
        mlflow.set_tracking_uri(cfg['location']['registry_uri']) 
        mlflow.set_experiment(self.experiment_name)
        for parameters_values in self.result:

            with mlflow.start_run():
                # testing the mlflow experiment id

                # setting the parameters to the given machine learning model
                self.model.set_params(**parameters_values)

                # creating the training dataframe
                train_x = self.train_data[0]
                train_y = self.train_data[1]

                # training the given model
                self.model.fit(train_x, train_y)

                # creating the testing dataframe
                test_x = self.test_data[0]
                test_y = self.test_data[1]

                # predicting the values for given test data
                y_ped = self.model.predict(test_x)

                # logging the data parameters
                mlflow.log_param('data_location', self.data_location)
                mlflow.log_param('data_version', self.data_version)
                mlflow.log_param('features_available', list(train_x.columns))
                mlflow.log_param('size_of_observations', train_x.shape[0])

                # logging the parameters
                for parameter, parameter_value in parameters_values.items():
                    mlflow.log_param(parameter, parameter_value)

               # getting the evaluation metrics for the training model
               # the problem type supported only 'classification', 'regression'  and  'clustering'
                output = EvalMetric(problem_type= self.problem_type , y_test=test_y, y_pred=y_ped, idealFlag=0, metricName=None, sample=None,beta=None,
                           pred_prob=None, average='weighted')
                # creating the output dictionary
                output_dictionary = output.to_dict('dict')
                metric_names = output_dictionary['Metrics']
                metric_values = output_dictionary[ 'Score']

                # logging the metrics
                for count in range(0, len(metric_names)):
                    mlflow.log_metric(metric_names[count],float(metric_values[count]))
                mlflow.sklearn.log_model(self.model,"model")


class SelectBestRun:
    ''' class for the selecting the best run from the experiment '''
    mlflow.set_tracking_uri(cfg['location']['registry_uri'])
    os.chdir(cfg['location']['mlflow_artifacts'])
    print('getting current directory after changing in the select best run', os.getcwd())
    def __init__(self, experiment_name):
        # initializing the variables
        os.chdir(cfg['location']['mlflow_artifacts'])
        print('getting current directory after changing in the select best run', os.getcwd())
        self.experiment_id = client.get_experiment_by_name(experiment_name).experiment_id
        self.experiment_results = mlflow.search_runs(experiment_ids=self.experiment_id)
        self.best_run= {}
        self.run_count = {}
        self.best_model_run_id = []

        # metrics which need to be minimum for best model
        self.min_metrics = ['metrics.Mean Absolute Error', 'metrics.Mean Squared Error', 'metrics.Root Mean Squared Log Error',
                            'metrics.Mean Absolute Percentage Error', 'metrics.Mean Squared Log Error', 'metrics.Root Mean Squared Error'
                            ,'metrics.Median Absolute Error', 'metrics.Max Error', 'metrics.Hamming Loss', 'metrics.Log Loss', 'metrics.Zero One Loss'
                            ,'metrics.Davies Bouldin Score']

        # metrics which need to be maximum for best model
        self.max_metrics = ['metrics.Explained Variance Score', 'metrics.R2 Score', 'metrics.Gini Score', 'metrics.Accuracy',
                            'metrics.Precision','metrics.Recall', 'metrics.F1 Score','metrics.F-Beta Score', 'metrics.AUC Score'
                            ,'metrics.Matthews CorrCoef', 'metrics.Cohen Kappa Score', 'metrics.Silhouette Score', 'metrics.Silhouette Sample'
                            ,'metrics.Mutual Info Score', 'metrics.Normalized Mutual Info Score', 'metrics.Adjusted Mutual Info Score',
                            'metrics.Adjusted Rand Score', 'metrics.Fowlkes Mallows Score', 'metrics.Homogeneity Score', 'metrics.Completeness Score'
                            , 'metrics.V Measure Score', 'metrics.Calinski Harabasz Score']

    # sub - function for the selecting best run based on the metric
    def best_metric_run(self,  metric_name):
        ''' selecting the best run for each given metric'''
        if metric_name in self.max_metrics:
            run_id = self.experiment_results['run_id'][self.experiment_results[metric_name] == self.experiment_results[metric_name].max()].iloc[0]
            self.best_run[metric_name] = run_id
        elif metric_name in self.min_metrics:
            run_id = self.experiment_results['run_id'][self.experiment_results[metric_name] == self.experiment_results[metric_name].min()].iloc[0]
            self.best_run[metric_name] = run_id

    # function for the selecting best run
    def best_run_id (self):
        ''' Taking the experimentation data selecting the best run based on the metrics info'''
        for column_name in list(self.experiment_results.columns):
            if column_name[:7] == 'metrics':
                self.best_metric_run( column_name)
        for run in self.best_run.values():
            if run in self.run_count.keys():
                self.run_count[run] = self.run_count[run] + 1
            if run not in self.run_count.keys():
                self.run_count[run] = 1
        sorted_run = sorted(self.run_count.items(), key=lambda kv: kv[1])
        self.best_model_run_id = sorted_run[0][0]
        print('returning best run id ', self.best_model_run_id)
        return self.best_model_run_id

    def register_model (self, model_name):
        ''' registering the model in the model registry'''
        mlflow.register_model(f"runs:/{self.best_model_run_id}/model", model_name)



class GettingModel:
    ''' Getting the selected model from the mlflow model registry based on the model name and model version'''
    def __init__(self, model_name,model_version ):
        self.model_name = model_name
        self.model_version = model_version
    def model(self):
        model = mlflow.pyfunc.load_model(
            model_uri=f"models:/{self.model_name}/{self.model_version}"
        )
        return model


class MultipleModel:

    ''' This class can be used for applying different sci-kit algorthims and logging the details in the mlflow'''
    def __init__(self,data_location, data_version, train_data,test_data,experiment_name, problem_type):
        self.train_data = train_data
        self.experiment_name = experiment_name
        self.test_data = test_data
        self.problem_type = problem_type
        self.data_location = data_location
        self.data_version = data_version
    def mlflow_run(self):
        '''Apply different algorthims to the data and logging the parameters and metrics for each algorthim'''

        # setting the tracking uri, this will change depends on the database
        # for postgres database : mlflow.set_tracking_uri('postgresql://postgres:postgres@localhost/mlflow')
        mlflow.set_tracking_uri(cfg['location']['registry_uri'])
        # setting the experiment name
        mlflow.set_experiment(self.experiment_name)
        if self.problem_type == 'regression':
            models_filter = 'regressor'
        if self.problem_type == 'classification':
            models_filter = 'classifier'
       # filtering the ml algorthims based upon the problem type  and storing in the variable 
        estimators = all_estimators(type_filter=models_filter)
        all_models = []
        for name, modelclass in estimators:
            try:
                model = modelclass()
                all_models.append(model)
            except Exception as e:
                print(e)
        # training the different algorthims to the same data and logging the parameters and metrics for each algorthim
        for self.model in all_models:
            with mlflow.start_run():
                try:
                    # creating the training dataframe
                    train_x = self.train_data[0]
                    train_y = self.train_data[1]

                    # training the given model
                    self.model.fit(train_x, train_y)

                    # creating the testing dataframe
                    test_x = self.test_data[0]
                    test_y = self.test_data[1]

                    # predicting the values for given test data
                    y_ped = self.model.predict(test_x)
                    try:
                        y_pred_prob =  self.model.predict_proba(test_x)
                    except:
                        y_pred_prob = None

                    # logging the data parameters
                    mlflow.log_param('data_location', self.data_location)
                    mlflow.log_param('data_version', self.data_version)
                    mlflow.log_param('features_available', list(train_x.columns))
                    mlflow.log_param('size_of_observations', train_x.shape[0])
                    mlflow.log_param('model_name', str(self.model))

                    # getting the evaluation metrics for the training model
                    # the problem type supported only 'classification', 'regression'  and  'clustering'
                    output = EvalMetric(problem_type=self.problem_type, y_test=test_y, y_pred=y_ped, idealFlag=0,
                                        metricName=None, sample=None, beta=0.5,
                                        pred_prob=y_pred_prob, average='weighted')
                    # creating the output dictionary
                    output_dictionary = output.to_dict('dict')
                    metric_names = output_dictionary['Metrics']
                    metric_values = output_dictionary['Score']
                    # logging the metrics
                    for count in range(0, len(metric_names)):
                        try:
                            mlflow.log_metric(metric_names[count],float(metric_values[count]))
                        except:
                            pass
                    # logging the model
                    mlflow.sklearn.log_model(self.model, "model")
                except Exception as e:
                    print(e)


class HyperoptModelSelection:
    ''' This class is having dependecy on the hyperopt_model_selection.py. Load the python file while executing the class.
        Here we are using 'hyperopt' Hyperparmeter Tuning library for fitting and tuning the ten algorthims 
        based on the problem type  regression or classification '''

    def __init__(self,data_location, data_version, train_data,test_data, experiment_name, problem_type):
        # intializing the parameters required
        self.train_data = train_data
        self.model = ''
        self.experiment_name = experiment_name
        self.test_data = test_data
        self.problem_type = problem_type
        self.data_location = data_location
        self.data_version = data_version

    def mlflow_runs(self):
        ''' performing the experiment run and storing the best model parameters and the metrics in the mlflow '''
        mlflow.set_tracking_uri(cfg['location']['registry_uri']) 
        mlflow.set_experiment(self.experiment_name)

        with mlflow.start_run():
            run = mlflow.active_run()
            print("Active run_id: {}".format(run.info.run_id))

            # creating the training dataframe
            train_x = self.train_data[0]
            train_y = self.train_data[1]

            # creating the testing dataframe
            test_x = self.test_data[0]
            test_y = self.test_data[1]
            # applying the hyperparamter tuning based upon problem type
            if self.problem_type == 'regression':
                parameters, self.model = regression_models(train_x,train_y)
                y_ped = self.model.predict(test_x)
                y_pred_prob = None
            if self.problem_type == 'classification':
                parameters, self.model = classification_models(train_x,train_y)
                y_ped = self.model.predict(test_x)
                try:
                   y_pred_prob =  self.model.predict_proba(test_x)
                except:
                   y_pred_prob = None
            # logging the data parameters
            mlflow.log_param('data_location', self.data_location)
            mlflow.log_param('data_version', self.data_version)
            mlflow.log_param('features_available', list(train_x.columns))
            mlflow.log_param('size_of_observations', train_x.shape[0])

            # logging the parameters
            for parameter, parameter_value in parameters.items():
                mlflow.log_param(parameter, parameter_value)

           # getting the evaluation metrics for the training model
           # the problem type supported only 'classification', 'regression'  and  'clustering'
            output = EvalMetric(problem_type= self.problem_type , y_test=test_y, y_pred=y_ped, idealFlag=0, metricName=None, sample=None,beta=0.5,
                       pred_prob=y_pred_prob, average='weighted')
            # creating the output dictionary
            output_dictionary = output.to_dict('dict')
            metric_names = output_dictionary['Metrics']
            metric_values = output_dictionary[ 'Score']

            # logging the metrics
            for count in range(0, len(metric_names)):
                mlflow.log_metric(metric_names[count],float(metric_values[count]))
            mlflow.sklearn.log_model(self.model, "model")
