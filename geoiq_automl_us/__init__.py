
"""
geoiq_automl_us
~~~~~~

The geoiq_automl_us package - a Python package project that is intended
to be used for creating models using geoiq automl platform.
"""

import os
import pandas as pd
import requests
import datetime
import matplotlib.pyplot as plt
import time
import ast
import json


class automl:
    """
    A Python package for creating models using the geoiq automl platform.

    This class provides methods for interacting with the geoiq automl API, including creating and managing datasets,
    training models, and making predictions.

    Parameters
    ----------
    authorization_key : str
        The authorization key for accessing the geoiq automl API.

    Attributes
    ----------
    headers : dict
        The headers used in API requests, including the authorization key and content type.

    Methods
    -------
    __init__(authorization_key)
        Constructor method for initializing the automl object.
        batch_prediction_download(prediction_id, download_path)
        Download the results of batch predictions as a CSV file.
    create_custom_model
        Create a custom model on a specific dataset with the given model parameters.
    create_dataset
        Create a new dataset using a CSV file.
    create_deployed_model_endpoint
        Create a deployed model endpoint for making predictions using a trained model.
    create_lift_chart
        Generate a lift chart for a trained model.
    create_roc_chart
        Generate a ROC chart for a trained model.
    create_validation_dataset
        Create a validation dataset for a specific dataset with the given validation percentage.
    create_validation_lift_chart
        Generate a lift chart for a trained model using a validation dataset.
    create_validation_roc_chart(model_id, validation_dataset_id)
        Generate a ROC chart for a trained model using a validation dataset.
    data_enrichment
        Perform data enrichment on a specific dataset.
    dataset_info
        Get information about a specific dataset.
    delete_dataset
        Delete a dataset.
    delete_model
        Delete a trained model.
    describe_dataset
        Get the progress status of dataset creation and models trained on the dataset.
    describe_model
        Get information about a specific trained model.
    eda
        Perform exploratory data analysis on a specific dataset.
    get_confusion_matrix
        Get the confusion matrix for a trained model.
    get_deployed_model_endpoint
        Get information about a deployed model endpoint.
    get_feature_importance
        Get the feature importance scores for a trained model.
    get_gains_table
        Get the gains table for a trained model.
    get_model_var
        Get the variable summary for a trained model.
    get_validation_gains_table
        Get the gains table for a trained model using a validation dataset.
    get_validation_psi_table
        Get the PSI (Population Stability Index) table for a trained model using a validation dataset.
    list_datasets
        List all the datasets created.
    list_models
        List all the models created on a specific dataset.
    list_validation_datasets
        List all the validation datasets created for a specific dataset.
    model_summary
        Get the summary of a specific trained model.
    variable_distribution_plot
        Generate a variable distribution plot for a specific dataset and variable.

    """
    def __init__(self, authorization_key):
        self.headers = {
            'x-api-key': authorization_key,
            'Content-Type': 'application/json'
        }
        now = datetime.datetime.now()
        print(f"AutoML object is created at {now}")
        
##### Data Upload API

    def create_dataset(self,df, dataset_name ='sample_data' ,dv_col = 'dv', dv_positive = '1',latitude_col = '' ,
          longitude_col = '',unique_col = 'geoiq_identifier_col',geocoding = 'false',
          address_col = '', pincode_col = '' , additional_vars = [] ):
        """
        Creates a dataset on the geoiq automl platform using the provided dataframe.
        
        Parameters
        ----------
        df : pandas DataFrame
            A dataframe containing the user data, which should contain the target variable, customer location details, and additional variables.
        dataset_name : str, optional
            Name of the dataset (default is 'sample_data').
        dv_col : str, optional
            Target variable column name (default is 'dv').
        dv_positive : str, optional
            Specify the positive observation in the dataset (default is '1').
        latitude_col : str, optional
            Latitude column in the dataframe (default is '').
        longitude_col : str, optional
            Longitude column in the dataframe (default is '').
        unique_col : str, optional
            Unique identifier in the dataset (default is 'geoiq_identifier_col').
        geocoding : str, optional
            If geocoding is required, set to 'true' (default is 'false').
        address_col : str, optional
            Address column in the dataframe (default is '').
        pincode_col : str, optional
            Pincode column in the dataframe (default is '').
        additional_vars : list, optional
            Additional variables to be passed in the creation of the model (default is []).
        
        Returns
        -------
        dataset_id : str
            Unique identifier for the dataset.
        """
        url = "https://automlapis-us.geoiq.io/wrapper/prod/dataset/v1.0/createdataset"
        
        

        df.to_csv(f'/tmp/{dataset_name}.csv',index=False)
        
#         self.headers = {
#           'x-api-key': s
#         }

        if (latitude_col == '')|(longitude_col == ''):
            payload={'dataset_name': dataset_name,
            'dv_col': dv_col,
            'dv_positive': dv_positive,
            'latitude':'""',
            'longitude':'""',
            'unique_col': unique_col,
            'geocoding': geocoding,
            'address': address_col,
            'pincode': pincode_col,
            'user_selected_vars': f'{additional_vars}'}
        elif (address_col == '')&(pincode_col == ''):
            payload={'dataset_name': dataset_name,
            'dv_col': dv_col,
            'dv_positive': dv_positive,
            'latitude':latitude_col,
            'longitude':longitude_col,
            'unique_col': unique_col,
            'geocoding': geocoding,
            'address': '\'\'',
            'pincode': '\'\'',
            'user_selected_vars': f'{additional_vars}'}
        elif (pincode_col == '')&(latitude_col == '')&(longitude_col == ''):
            payload={'dataset_name': dataset_name,
            'dv_col': dv_col,
            'dv_positive': dv_positive,
            'latitude':'""',
            'longitude':'""',
            'unique_col': unique_col,
            'geocoding': geocoding,
            'address': address_col,
            'pincode': '\'\'',
            'user_selected_vars': f'{additional_vars}'}

        
        files=[
          ('df_data',(f'{dataset_name}.csv',open(f'/tmp/{dataset_name}.csv','rb'),'text/csv'))
        ]
        headers = {'x-api-key' : self.headers['x-api-key']}
        response = requests.request("POST", url, headers=headers, data=payload, files=files)
    
        os.remove(f'/tmp/{dataset_name}.csv')
        
        print(response.text)
        dataset_id = response.json()['data']['dataset_id']
                
        return dataset_id
    
    
##### Data Enrichment


    def data_enrichment(self,dataset_id, var_list):
        """
        Enriches the data in the dataset with GeoIQ variables and returns a download URL.

        Parameters
        ----------
        dataset_id : str
            Unique identifier for the dataset.
        var_list : list
            List of variables to be extracted.

        Returns
        -------
        str
            A URL through which enriched data containing GeoIQ variables can be downloaded.


        Notes
        -----
        This function makes a POST request to the GeoIQ API to enrich the data in the dataset with
        the specified GeoIQ variables. The dataset_id parameter should be a valid identifier for
        an existing dataset in the GeoIQ system. The var_list parameter should be a list of
        variable names to be extracted. The function returns a download URL for the enriched data
        in JSON format.

        Examples
        --------
        >>> dataset_id = "12345"
        >>> var_list = ["var1", "var2", "var3"]
        >>> url = self.data_enrichment(dataset_id, var_list)
    

        """
        url = "https://automlapis-us.geoiq.io/wrapper/prod/dataset/v1.0/downloadgeoiqvarsdata"

        payload = json.dumps({
          "dataset_id": dataset_id,
          "var_list": str(var_list)
        })


        response = requests.request("POST", url, headers=self.headers, data=payload)
        
        if (len(response.json())==3):
            print(response.json()['download_url'])
        if (len(response.json())==1):
            print(response.json()['data']['message'])
        

        
        
#### Describe Dataset


    def describe_dataset(self,dataset_id):
        """
        Retrieve the progress status details of the dataset creation process for the given dataset identifier.

        Parameters
        ----------
        dataset_id : str
            Unique identifier for the dataset.

        Returns
        -------
        dict
            A dictionary containing the progress status details of the dataset creation process. If the dataset
            creation is complete, it also includes the details of the models created on the dataset.
        """
        
        url = "https://automlapis-us.geoiq.io/wrapper/prod/progress/v1.0/getdatasetprogress"
        payload = json.dumps({
          "dataset_id": dataset_id
        })

        response = requests.request("POST", url, headers=self.headers, data=payload)
        
        if (response.json()['data']['progress'][0]['status_text']) == 'complete':
            url = "https://automlapis-us.geoiq.io/wrapper/prod/model/v1.0/getalldatasetmodels"

            payload = json.dumps({
              "dataset_id": dataset_id
            })

            response = requests.request("GET", url, headers=self.headers, data=payload)
            print(f"Dataset creation is completed for this dataset id {dataset_id}")
            return response.json()['data']
        else:
            progress_dict  = response.json()['data']['progress'][0]
            return {key: progress_dict[key] for key in ['dataset_id','dataset_name','status_text']}
            

    
    
## MODEL GET ALL DATASET


    def list_datasets(self):
        """
        Lists all the datasets created by the user.

        Parameters
        ----------
        None

        Returns
        -------
        pandas.DataFrame
            DataFrame containing information about all the datasets created by the user.
        """
        
        url = "https://automlapis-us.geoiq.io/wrapper/prod/dataset/v1.0/getalldataset"

        payload={}

        response = requests.request("POST", url, headers=self.headers, data=payload)
        y = pd.DataFrame(response.json()['data'])
        
#         y.style.set_properties(**{'url': self.make_clickable})
#         print(y)
        return y
    
## GET ALL DATASET MODELS

    def list_models(self,dataset_id):
        """
        Retrieve a list of all the models created on the given dataset.

        Parameters
        ----------
        dataset_id : str
            Unique identifier for the dataset.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing information about all the models created on the given dataset.
        """
        
        url = "https://automlapis-us.geoiq.io/wrapper/prod/model/v1.0/getalldatasetmodels"

        payload = json.dumps({
          "dataset_id": dataset_id
        })


        response = requests.request("GET", url, headers=self.headers, data=payload)
        
        y = pd.DataFrame(response.json()['data'])
        return y


        
##### EDA API

    def eda(self,dataset_id):
        """
        Performs Exploratory Data Analysis (EDA) on a given dataset and returns the results as a Pandas DataFrame.

        Parameters
        ----------
        dataset_id : str
            A unique identifier for the dataset to perform EDA on.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the results of the EDA, including various statistical measures and information
            about the dataset's columns.


        Notes
        -----
        This function makes a POST request to the GeoIQ API to perform EDA on the given dataset. The dataset_id
        parameter should be a valid identifier for an existing dataset in the GeoIQ system. The function returns
        the results of the EDA in the form of a Pandas DataFrame, which includes information such as column names,
        column types, IV (Information Value), AUC (Area Under Curve), bins, catchment, category, F-test p-value,
        T-test p-value, description, deviation, direction, ID, KS (Kolmogorov-Smirnov statistic), major category,
        max KS, mean, name, normalization level, ROC (Receiver Operating Characteristic) curve, standard deviation,
        sub-category name, unique count, and variable. The DataFrame is sorted by IV in descending order.

        Examples
        --------
        >>> dataset_id = "12345"
        >>> df_eda = self.eda(dataset_id)
        >>> print(df_eda)
                    column_name  column_type  iv  auc_1  auc_2  auc_3  bins  catchment  \
        0             var1         numeric  0.5   0.85   0.90   0.95    10       True   
        1             var2         categorical  0.4   0.75   0.80   0.85     5       True   
        2             var3         numeric  0.3   0.60   0.70   0.80    20       False   
        ...           ...              ...  ...    ...    ...    ...   ...        ...

        """
#         params = {
#             'dataset_id': dataset_id,
#         }
#         response = requests.get('https://automlapis.geoiq.io/cloudmlapis/column/v1.0/getcolumnstatscategorydata', params=params, headers=self.headers)
        url = "https://automlapis-us.geoiq.io/wrapper/prod/dataset/v1.0/getdataseteda"

        payload = json.dumps({
          "dataset_id": dataset_id
        })


        response = requests.request("POST", url, headers=self.headers, data=payload)
    
        df_nested_list = pd.json_normalize(response.json())

        return pd.DataFrame(df_nested_list['data.data.geoiq_vars'][0])[[ 'column_name', 'column_type','iv','auc_1', 'auc_2', 'auc_3', 'bins',
       'catchment', 'category', 'F_test_pvalue', 'T_test_pvalue', 
       'desc_name', 'description', 'deviation', 'direction', 'id', 'ks',
       'major_category', 'max_ks', 'mean', 'name', 'normalization_level',
        'roc', 'sd', 'sub_category_name', 'unique',
       'unique_count', 'variable']].sort_values('iv',ascending=False).reset_index(drop=True)

#### Datasetinfo


    def dataset_info(self,dataset_id):
        """
        Retrieve information about the dataset with the given unique identifier.

        Parameters
        ----------
        dataset_id : str
            Unique identifier for the dataset.

        Returns
        -------
        dict
            A dictionary containing information about the dataset, including its creation and update timestamps.
        """
        
        url = "https://automlapis-us.geoiq.io/wrapper/prod/dataset/v1.0/getdatasetinfo"

        payload = json.dumps({
          "dataset_id": dataset_id
        })


        response = requests.request("POST", url, headers=self.headers, data=payload)
        
        dict_nested = response.json()['data']['data']
            
        dict_nested['created_at'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(dict_nested['created_at']))
        dict_nested['updated_at'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(dict_nested['updated_at']))
    
        return {key: dict_nested[key] for key in ['address_col', 'created_at', 'data_path', 'data_type', 'dv_col',
                                     'dv_positive', 'dv_rate', 'geocoding', 'identifier_col',
                                      'lat_col', 'lng_col', 'name', 'number_of_categorical',
                                     'number_of_columns', 'number_of_numerical', 'number_of_rows', 'number_of_rows_geoiq',
                                     'pincode_col', 'remarks', 'status', 'total_dv_rate', 'updated_at', 'user_selected_vars']}

    
    def delete_dataset(self, dataset_id):
        
        """
        Delete a dataset with the given dataset identifier.

        Parameters
        ----------
        dataset_id : str
            Unique identifier for the dataset to be deleted.

        Returns
        -------
        None
            This function does not return any value. It prints a message indicating whether the dataset was
            deleted successfully or not.
        """
        
        url = "https://automlapis-us.geoiq.io/wrapper/prod/dataset/v1.0/deletedataset"

        payload = json.dumps({
          "dataset_id": dataset_id
        })

        response = requests.request("POST", url, headers=self.headers, data=payload)
        if (response.json()['data']['message'] == 'success'):
            print('Dataset deleted successfully')
        else:
            print('Dataset is not available')
        

#### Chart API


    def variable_distribution_plot(self,dataset_id,variable_name, quantize=True):
        """
        Plots the distribution of a variable from a given dataset using quantize or quantile data.

        Parameters
        ----------
        dataset_id : str
            An unique identifier for the dataset.
        variable_name : str
            The name of the variable to be plotted.
        quantize : bool, optional
            Determines whether to use quantize data or quantile data for plotting. 
            If True, uses quantize data. If False, uses quantile data. 
            (Default value is True)

        Returns
        -------
        None
            Plots the variable distribution as a bar chart with count of observations on the primary y-axis 
        and percentage of positive observations on the secondary y-axis.

        """
        
        url = "https://automlapis-us.geoiq.io/wrapper/prod/dataset/v1.0/getdataseteda"

        payload = json.dumps({
          "dataset_id": dataset_id
        })


        response = requests.request("POST", url, headers=self.headers, data=payload)
    
        df_nested_list = pd.json_normalize(response.json())
        df_nested_list = pd.DataFrame(df_nested_list['data.data.geoiq_vars'][0])
        
        xlabel = df_nested_list[df_nested_list['column_name']==variable_name]['description'].values[0]
        
        ### Graph API
        
        url = "https://automlapis-us.geoiq.io/wrapper/prod/column/v1.0/getcolumnfrequencygraph"
        
        col_id =  df_nested_list[df_nested_list['column_name']==variable_name]['id'].values[0]

        payload = json.dumps({
          "dataset_id": dataset_id,
          "column_id": col_id
        })


        response = requests.request("POST", url, headers=self.headers, data=payload)
        if quantize ==True:
            df_nested_list = pd.json_normalize(response.json()['data']['quantize_data'])
        else:
            df_nested_list = pd.json_normalize(response.json()['data']['quantile_data'])
            
        fig = plt.figure(figsize=(10,5))
        ax1 = fig.add_subplot()
        if quantize ==True:
            ax1.set_title('EDA quantize chart')
        else:
            ax1.set_title('EDA quantile chart')

        ax2 = ax1.twinx()
        ax1.bar(df_nested_list['range'],df_nested_list['count'],label="Count", color='r')
        ax2.plot(df_nested_list['range'],df_nested_list['avg_dv'],label="% of positive observation", color='b')

        fig.autofmt_xdate(rotation=45)
        # ax1.set_xticklabels(ax1.get_xticklabels(),rotation=45)
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel('Number of rows', color='r')
        ax2.set_ylabel('% of positive observation', color='b')
        fig.legend(loc=1)
        plt.show()


##### Custom Model
    def create_custom_model(self,dataset_id,model_name,model_type = "xgboost", split_ratio ="[0.7,0.3,None]"):
        """
        Creates a custom machine learning model for a given dataset.

        Parameters
        ----------
        dataset_id : str
            Unique identifier for the dataset.
        model_name : str
            Name of the model.
        model_type : str, optional
            Type of the model to be created. Must be one of ['logistic', 'xgboost']. 
            (Default value is "xgboost")
        split_ratio : str, optional
            Train-Validation-Test ratio as a list of three values: [train_ratio, validation_ratio, test_ratio].
            (Default value is "[0.7,0.3,None]")

        Returns
        -------
        str
            Model ID of the created model.

        """
        url = "https://automlapis-us.geoiq.io/wrapper/prod/model/v1.0/createmodel"

        payload = json.dumps({
          "dataset_id": dataset_id,
          "model_name": model_name,
          "model_type": model_type,
          "split_ratio": split_ratio
        })
 

        response = requests.request("POST", url, headers=self.headers, data=payload)
        model_id = ast.literal_eval(response.text)['data']['model_id']
        

        return model_id
    
    def delete_model(self, model_id):
        """
        Deletes a trained model from the GeoIQ AutoML API by its unique ID.

        Parameters
        ----------
            model_id (str): The ID of the model to delete.

        Returns
        -------
            None

        """

        url = "https://automlapis-us.geoiq.io/wrapper/prod/dataset/v1.0/deletemodel"

        payload = json.dumps({
          "model_id": model_id
        })

        response = requests.request("POST", url, headers=self.headers, data=payload)
        if (response.json()['data']['message'] == 'success'):
            print('Model deleted successfully')
        else:
            print('Model is not available')
    
        
#### Describe Model


    def describe_model(self,model_id):
        """

        Parameters
        ----------
        model_id : An unique identifier for the model
            

        Returns
        -------
        Return a dict containing the model progress status details
        

        """
        url = "https://automlapis-us.geoiq.io/wrapper/prod/progress/v1.0/getmodelprogress"
        payload = json.dumps({
          "model_id": model_id
        })

        response = requests.request("POST", url, headers=self.headers, data=payload)
        progress_dict  = response.json()['data']['progress'][0]
        return {key: progress_dict[key] for key in ['model_id','model_name','status_text']} 



#### Model Details

    def model_summary(self,model_id):
        """
        Retrieves performance metrics and other details of a machine learning model by its ID.

        Parameters
        ----------
        model_id : str
            Unique identifier for the model.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing performance metrics and other details of the model, including model name,
            start time, completion time, dependent variable column name, dependent variable rate, model comments,
            variable count, IV (Information Value) for holdout data, IV for training data, holdout AUC (Area Under
            the Curve), training AUC, holdout KS (Kolmogorov-Smirnov statistic), and training KS.

        """
        
        url = "https://automlapis-us.geoiq.io/wrapper/prod/model/v1.0/getmodeloverview"

        payload = json.dumps({
          "model_id": model_id
        })


        response = requests.request("GET", url, headers=self.headers, data=payload)
        df_nested_list = pd.DataFrame([response.json()['data']])
        df_nested_list['completed_at'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(df_nested_list['completed_at']))
        df_nested_list['started_at'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(df_nested_list['started_at']))
        
        return df_nested_list[['name','started_at','completed_at','dv_col', 'dv_rate','model_comments', 'variable_count',
                        'iv', 'iv_train','holdout_auc','train_auc', 'holdout_ks', 'train_ks']].rename(columns={'iv':'iv_holdout'})

        


## LIFT Chart

    def create_lift_chart(self,model_id):
        """
        Generates a lift chart for the given model ID.

        Parameters
        ----------
        model_id : str
            An unique identifier for the model.

        Returns
        -------
        None
            Plots the lift chart.
        """
        
        url = "https://automlapis-us.geoiq.io/wrapper/prod/model/v1.0/getmodelliftchart"

        payload = json.dumps({
          "model_id": model_id
        })


        response = requests.request("GET", url, headers=self.headers, data=payload)

        df_nested_list = pd.DataFrame(response.json()['data'])
        plt.title('Lift chart')
        plt.plot(df_nested_list['x'],df_nested_list['holdout_decile'],label="holdout_decile")
        plt.plot(df_nested_list['x'],df_nested_list['train_decile'],label="train_decile")
        plt.ylabel('Average target value')
        plt.xlabel('Bins based on predicted value')
        plt.legend(loc=4)
        
    

##### ROC Chart

    def create_roc_chart(self,model_id):
        """
        Generates a ROC (Receiver Operating Characteristic) curve for the given model ID.

        Parameters
        ----------
        model_id : str
            An unique identifier for the model.

        Returns
        -------
            Plots the ROC curve.
        """
        
        
        ## AUC values
        url = "https://automlapis-us.geoiq.io/wrapper/prod/model/v1.0/getmodeloverview"

        payload = json.dumps({
          "model_id": model_id
        })


        response = requests.request("GET", url, headers=self.headers, data=payload)

        df_nested_list1 = pd.json_normalize(response.json()['data'])
        
        ## Roc
        
        url = "https://automlapis-us.geoiq.io/wrapper/prod/model/v1.0/getmodelroccurve"


        response = requests.request("GET", url, headers=self.headers, data=payload)
        df_nested_list = pd.DataFrame(response.json()['data'])

        plt.title('ROC Curve')
        plt.plot(df_nested_list['fpr'],df_nested_list['tpr_train'],label="Train AUC="+str(df_nested_list1['train_auc'][0]))
        plt.plot(df_nested_list['fpr'],df_nested_list['tpr_holdout'],label="Holdout AUC="+str(df_nested_list1['holdout_auc'][0]))
        plt.ylabel('True positive rate (Sensitivity)')
        plt.xlabel('False positive rate (Fallout)')
        plt.legend(loc=4)


##### Feature Importance

    def get_feature_importance(self,model_id):
        """
        Retrieves the feature importance for the given model ID.

        Parameters
        ----------
        model_id : str
            An unique identifier for the model.

        Returns
        -------
        pd.DataFrame
            A dataframe containing the feature importance.
        """
        
        url = "https://automlapis-us.geoiq.io/wrapper/prod/model/v1.0/getmodelfeatureimportance"

        payload = json.dumps({
          "model_id": model_id
        })

        response = requests.request("GET", url, headers=self.headers, data=payload)

        return pd.json_normalize(response.json()['data'])


##### Gain Table

    def get_gains_table(self,model_id,split = 'train'):
        """
        Retrieves the gains table for a given model ID.

        Parameters
        ----------
        model_id : str
            A unique identifier for the model.
        split : str, optional
            The split for which to retrieve the gains table, either 'train' or 'holdout'.
            (default is 'train')

        Returns
        -------
        pd.DataFrame
            A pandas DataFrame containing the gains table with the following columns:
            - 'Decile': Decile values
            - 'Score Range': Score range for each decile
            - 'Non-positive Class Count': Count of non-positive class instances in each decile
            - 'Positive Class Count': Count of positive class instances in each decile
            - 'KS': Kolmogorov-Smirnov (KS) statistic for each decile
            - 'Positive Class Percentage': Percentage of positive class instances in each decile


        Example
        -------
        >>> automl_model = AutoMLModel()
        >>> gains_table_train = automl_model.get_gains_table(model_id='model123', split='train')
        >>> gains_table_holdout = automl_model.get_gains_table(model_id='model123', split='holdout')

        """
        
        url = "https://automlapis-us.geoiq.io/wrapper/prod/model/v1.0/getgainstable"

        payload = json.dumps({
          "model_id": model_id
        })

        response = requests.request("GET", url, headers=self.headers, data=payload)
        
        if split == 'train':
            df_nested_list = pd.json_normalize(response.json()['data'][split]['data'])
            df_nested_list.rename(columns={'abs_diff_array_train':'KS', 'bucket_data_array':'Score Range', 'decile_data_array':'Decile',
       'percentage_data_array_train':'Positive Class Percentage', 'target_0_array_train':'Non-positive Class Count',
       'target_array_train':'Positive Class Count'},inplace=True)

            df_nested_list = df_nested_list[['Decile', 'Score Range', 'Non-positive Class Count', 'Positive Class Count','KS','Positive Class Percentage']]
        
        elif split == 'holdout':
            df_nested_list = pd.json_normalize(response.json()['data'][split]['data'])
            df_nested_list.rename(columns={'abs_diff_array_holdout':'KS', 'bucket_data_array':'Score Range', 'decile_data_array':'Decile',
       'percentage_data_array_holdout':'Positive Class Percentage', 'target_0_array_holdout':'Non-positive Class Count',
       'target_array_holdout':'Positive Class Count'},inplace=True)

            df_nested_list = df_nested_list[['Decile', 'Score Range', 'Non-positive Class Count', 'Positive Class Count','KS','Positive Class Percentage']]
        
        else:
            print(" Pass correct value in the split argument")
            
        return df_nested_list



##### Confusion Matrix

    def get_confusion_matrix(self,dataset_id,model_id,threshold):
        """
        Retrieves the confusion matrix for the given dataset ID, model ID, and threshold.

        Parameters
        ----------
        dataset_id : str
            An unique identifier for the dataset.
        model_id : str
            An unique identifier for the model.
        threshold : float
            The threshold used for prediction.

        Returns
        -------
        pd.DataFrame
            A dataframe containing the confusion matrix.
        """
        
        url = "https://automlapis-us.geoiq.io/wrapper/prod/model/v1.0/getmodelconfusionmatrix"

        payload = json.dumps({
        'model_id': model_id,
        'dataset_id': dataset_id,
        'threshold': str(threshold),
        })

        response = requests.request("GET", url, headers=self.headers, data=payload)
        return pd.json_normalize(response.json()['data'])

    def create_validation_dataset(self,df,model_id,dataset_id,name ='validation_dataset' ,dv_col = '\'\'', dv_positive = '\'\'',
                                       latitude = '\'\'' , longitude = '\'\'',unique_col = 'geoiq_identifier_col',
                                       geocoding = 'f', address= '\'\'', pincode= '\'\'',user_selected_vars = '\'\''):
        """
        Creates a validation dataset for the given model ID and dataset ID using the provided dataframe.

        Parameters
        ----------
        df : pd.DataFrame
            A dataframe containing the user data, which should contain the target variable, customer location details,
            and additional variables.
        model_id : str
            An unique identifier for the model.
        dataset_id : str
            An unique identifier for the dataset.
        name : str, optional
            Name of the validation dataset. (Default value is 'validation_dataset')
        dv_col : str, optional
            Target variable column name. (Default value is 'dv')
        dv_positive : str, optional
            Specify the positive observation in the dataset. (Default value is '1')
        latitude : str, optional
            Latitude column in the dataframe. (Default value is 'geo_latitude')
        longitude : str, optional
            Longitude column in the dataframe. (Default value is 'geo_longitude')
        unique_col : str, optional
            Unique identifier in the dataset. (Default value is 'geoiq_identifier_col')
        geocoding : str, optional
            If geocoding is required, it should be set to 'true'. (Default value is 'false')
        address : str, optional
            Address column in the dataframe. (Default value is '\'\'')
        pincode : str, optional
            Pincode column in the dataframe. (Default value is '\'\'')
        user_selected_vars : str, optional
            Additional variables to be passed in the creation of the model. (Default value is '\'\'')

        Returns
        -------
        str
            Unique identifier for the validation dataset.
        """
        
        url = "https://automlapis-us.geoiq.io/wrapper/prod/validationdatasets/v1.0/createvddataset"

        df.to_csv(f'/tmp/{name}.csv',index=False)

        #         headers = {
        #           'x-api-key': s
        #         }

        payload={'name': name,
        'dv_col': dv_col,
        'dv_positive': dv_positive,
        'latitude': latitude,
        'longitude': longitude,
        'unique_col': unique_col,
        'geocoding': geocoding,
        'address': address,
        'pincode': pincode,
        'model_id': model_id,
        'dataset_id': dataset_id,
        'user_selected_vars': user_selected_vars}

        files=[('df_data',(f'{name}.csv',open(f'/tmp/{name}.csv','rb'),'text/csv'))]
        headers = {'x-api-key' : self.headers['x-api-key']}
 
        response = requests.request("POST", url, headers=headers, data=payload, files=files)

        os.remove(f'/tmp/{name}.csv')

        print(response.text)
        dataset_id = response.json()['data']['validation_dataset_id']

        return dataset_id


    ## LIFT Chart

    def create_validation_lift_chart(self,validation_dataset_id):
        """
        Retrieves the lift chart for the given validation dataset ID.

        Parameters
        ----------
        validation_dataset_id : str
            Unique identifier for the validation dataset.

        Returns
        -------
            Plots the lift chart.
        """
        
        url = "https://automlapis-us.geoiq.io/wrapper/prod/validationdatasets/v1.0/getmodelliftchart"

        payload = json.dumps({
          "validation_dataset_id": validation_dataset_id
        })


        response = requests.request("GET", url, headers=self.headers, data=payload)

        df_nested_list = pd.DataFrame(response.json()['data'])
        plt.title('Lift chart')
        plt.plot(df_nested_list['x'],df_nested_list['validation_decile'],label="validation_decile")
        plt.plot(df_nested_list['x'],df_nested_list['train_decile'],label="train_decile")
        plt.ylabel('Average target value')
        plt.xlabel('Bins based on predicted value')
        plt.legend(loc=4)
        return


    ##### ROC Chart

    def create_validation_roc_chart(self,validation_dataset_id):
        """
        Creates a Receiver Operating Characteristic (ROC) chart for a given validation dataset.

        Parameters
        ----------
        validation_dataset_id : str
            Unique identifier for the validation dataset.

        Returns
        -------
        None
            Plots the ROC Curve.
        """


        ## AUC values
        url = "https://automlapis-us.geoiq.io/wrapper/prod/validationdatasets/v1.0/getvdevaluate"

        payload = json.dumps({
          "validation_dataset_id": validation_dataset_id
        })


        response = requests.request("POST", url, headers=self.headers, data=payload)

        df_nested_list1 = pd.json_normalize(response.json()['data'])

        ## Roc

        url = "https://automlapis-us.geoiq.io/wrapper/prod/validationdatasets/v1.0/getmodelroccurve"


        response = requests.request("GET", url, headers=self.headers, data=payload)
        df_nested_list = pd.DataFrame(response.json()['data'])

        plt.title('ROC Curve')
        plt.plot(df_nested_list['fpr'],df_nested_list['tpr_train'],label="Train AUC="+str(df_nested_list1['train_auc'][0]))
        plt.plot(df_nested_list['fpr'],df_nested_list['tpr_validation'],label="validation AUC="+str(df_nested_list1['validation_auc'][0]))
        plt.ylabel('True positive rate (Sensitivity)')
        plt.xlabel('False positive rate (Fallout)')
        plt.legend(loc=4)




    ##### Validation Gain Table

    def get_validation_gains_table(self,validation_dataset_id,split = 'train'):
        """
        Retrieves a validation gains table for a given validation dataset.

        Parameters
        ----------
        validation_dataset_id : str
            Unique identifier for the validation dataset.
        split : str, optional
            Split type ('train' or 'validation'), by default 'train'.

        Returns
        -------
        pandas.DataFrame
            Validation gains table as a DataFrame.
        """
        
        url = "https://automlapis-us.geoiq.io/wrapper/prod/validationdatasets/v1.0/getgainstable"

        payload = json.dumps({
          "validation_dataset_id": validation_dataset_id
        })

        response = requests.request("GET", url, headers=self.headers, data=payload)

        if split == 'train':
            df_nested_list = pd.json_normalize(response.json()['data'][split]['data'])
            df_nested_list.rename(columns={'abs_diff_array_train':'KS', 'bucket_data_array':'Score Range', 'decile_data_array':'Decile',
       'percentage_data_array_train':'Positive Class Percentage', 'target_0_array_train':'Non-positive Class Count',
       'target_array_train':'Positive Class Count'},inplace=True)

            df_nested_list = df_nested_list[['Decile', 'Score Range', 'Non-positive Class Count', 'Positive Class Count','KS','Positive Class Percentage']]

        elif split == 'validation':
            df_nested_list = pd.json_normalize(response.json()['data'][split]['data'])
            df_nested_list.rename(columns={'abs_diff_array_validation':'KS', 'bucket_data_array':'Score Range', 'validation_data_array':'Decile',
       'percentage_data_array_validation':'Positive Class Percentage', 'target_0_array_validation':'Non-positive Class Count',
       'target_array_validation':'Positive Class Count'},inplace=True)

            df_nested_list = df_nested_list[['Decile', 'Score Range', 'Non-positive Class Count', 'Positive Class Count','KS','Positive Class Percentage']]

        else:
            print(" Pass correct value in the split argument")

        return df_nested_list

    ##### Validation PSI Table

    def get_validation_psi_table(self,validation_dataset_id):
        """
        Retrieves a validation PSI (Population Stability Index) table for a given validation dataset.

        Parameters
        ----------
        validation_dataset_id : str
            Unique identifier for the validation dataset.

        Returns
        -------
        pandas.DataFrame
            Validation PSI table as a DataFrame.
        """
        
        url = "https://automlapis-us.geoiq.io/wrapper/prod/validationdatasets/v1.0/getpsitable"

        payload = json.dumps({
          "validation_dataset_id": validation_dataset_id
        })

        response = requests.request("GET", url, headers=self.headers, data=payload)
        df_nested_list = pd.DataFrame(response.json()['data']['train']['data'])

        return df_nested_list

    
    
    def batch_prediction_download(self, validation_dataset_id):
        """
        Downloads batch predictions for a given validation dataset.

        Parameters
        ----------
        validation_dataset_id : str
            Unique identifier for the validation dataset.

        Returns
        -------
        str
            URL to download the batch predictions.
        """
            
        url = "https://automlapis-us.geoiq.io/wrapper/prod/validationdatasets/v1.0/getvdproperties"

        payload = json.dumps({
          "validation_dataset_id": validation_dataset_id
        })

        response = requests.request("GET", url, headers=self.headers, data=payload)
        return response.json()['data']['prediction_file']
    
    ## MODEL GET ALL DATASET


    def list_validation_datasets(self,model_id):
        """
        Lists all the validation datasets created for a given model.

        Parameters
        ----------
        model_id : str
            Unique identifier for the model.

        Returns
        -------
        pandas.DataFrame
            List of validation datasets as a DataFrame.
        """
        
        url = "https://automlapis-us.geoiq.io/wrapper/prod/validationdatasets/v1.0/getallvddatasets"

        payload=json.dumps({
    "model_id": model_id
    })

        response = requests.request("GET", url, headers=self.headers, data=payload)
        y = pd.DataFrame(response.json()['data'])

    #         y.style.set_properties(**{'url': self.make_clickable})
    #         print(y)
        return y



    def get_deployed_model_endpoint(self,model_id):
        """
        Retrieves the endpoint of a deployed model for a given model ID.

        Parameters
        ----------
        model_id : str
            Unique identifier for the model.

        Returns
        -------
        str
            Endpoint URL of the deployed model.
        """
        
        url = "https://automlapis-us.geoiq.io/wrapper/prod/model/v1.0/getdeployedmodelendpoint"

        payload = json.dumps({
        "model_id": model_id
        })


        response = requests.request("GET", url, headers=self.headers, data=payload)
        if len(response.json()['data']) ==0:
            print('Model is not deployed yet')
        else:
            df_nested_list = pd.json_normalize(response.json()['data'])
            df_nested_list['created_at'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(df_nested_list['created_at']))
            return df_nested_list



    # Create Endpoint for Model / Deploy Model
    def create_deployed_model_endpoint(self,model_id):
        """
        Creates an endpoint for a deployed model.

        Parameters
        ----------
        model_id : str
            Unique identifier for the model.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing endpoint details, including the created_at timestamp.
        """
        
        url = "https://automlapis-us.geoiq.io/wrapper/prod/model/v1.0/createmodelendpoint"

        payload = json.dumps({
        "model_id": model_id
        })


        response = requests.request("POST", url, headers=self.headers, data=payload)
        df_nested_list = pd.json_normalize(response.json()['data'])
        df_nested_list['created_at'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(df_nested_list['created_at']))

        return df_nested_list

    
    def get_model_var(self, latitude,longitude, model_id):
        """
        Retrieves geoiq variables for a given latitude and longitude using a deployed model.

        Parameters
        ----------
        latitude : float
            Latitude value.
        longitude : float
            Longitude value.
        model_id : str
            Unique identifier for the model.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing geoiq variables for the given latitude and longitude.
        """
        
        url = "https://automlapis-us.geoiq.io/wrapper/prod/model/v1.0/getmodelfeatureimportance"

        payload = json.dumps({
          "model_id": model_id
        })
        response = requests.request("GET", url, headers=self.headers, data=payload)

        varlist = pd.json_normalize(response.json()['data'])['column_name'].tolist()

        url = "https://dataserving-us.geoiq.io/production/v1.0/getvariablesbulk"


        payload = json.dumps({
          "lat": latitude,
          "lng": longitude,
          "variables": ','.join(varlist)
        })


        response = requests.request("POST", url, headers=self.headers, data=payload)

        return pd.json_normalize(response.json()['data'])

