


"""
geoiq_automl
~~~~~~

The geoiq_automl package - a Python package project that is intended
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
    """ """
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

        Parameters
        ----------
        df : A dataframe containing the user data, which should contains target variable, customer location details, and additional variables. 
            
        dataset_name : Name of the dataset
             (Default value = 'test')
        dv_col : Target variable column name
             (Default value = 'dv')
        dv_positive : Specify the good observation in the dataset 
             (Default value = '1')
        latitude_col : Latitutde column in the dataframe
             (Default value = "geo_latitude")
        longitude_col : Latitutde column in the dataframe
             (Default value = "geo_longitude")
        unique_col : Unique identifier in the dataset
             (Default value = 'geoiq_identifier_col')
        geocoding : If geocoding is required need to be true
             (Default value = 'false')
        address_col : Address column in the dataframe
             (Default value = '\'\'')
        pincode_col : Pincode column in the dataframe
             (Default value = '\'\'')
        additional_vars : Additional variables need to be passed in the creation of the model
             (Default value = '[]')

        Returns
        -------

        """
        url = "https://automlapis-us.geoiq.io/wrapper/prod/dataset/v1.0/createdataset"
        
        

        df.to_csv(f'/tmp/{dataset_name}.csv',index=False)
        
#         self.headers = {
#           'x-api-key': s
#         }

        if (latitude_col = '')|(longitude_col = ''):
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
        elif (address = '')|(pincode = ''):
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

        Parameters
        ----------
        dataset_id : An unique identifier for the dataset
            
        var_list : List of variables need to be extracted
            

        Returns
        -------
        A url through which enriched data conatining geoiq variables can be downloaded. 
        

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

        Parameters
        ----------
        dataset_id : An unique identifier for the dataset
            

        Returns
        -------
        Return a dict containing the dataset creation progress status details

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
            return response.json()['data'][0]['model_id']
        else:
            return (response.json()['data']['progress'][0])
            

    
    
## MODEL GET ALL DATASET


    def list_datasets(self):
        """

        Parameters
        ----------
        No Parameters required
            

        Returns
        -------
        Lists alll the datasets created by the user
        

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

        Parameters
        ----------
        dataset_id : An unique identifier for the dataset
            

        Returns
        -------
        Lists alll the models created on the given dataset

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

        Parameters
        ----------
        dataset_id :  An unique identifier for the dataset
            

        Returns
        -------
        Returns the dataframe containing exploratory data analysis on the given dataframe


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

        Parameters
        ----------
        dataset_id :  An unique identifier for the dataset
            

        Returns
        -------

        """
        
        url = "https://automlapis-us.geoiq.io/wrapper/prod/dataset/v1.0/getdatasetinfo"

        payload = json.dumps({
          "dataset_id": dataset_id
        })


        response = requests.request("POST", url, headers=self.headers, data=payload)
        
        dict_nested = response.json()['data']['data']
            
        dict_nested['created_at'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(dict_nested['created_at']))
        dict_nested['updated_at'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(dict_nested['updated_at']))
    
        return dict_nested

    
    def delete_dataset(self, dataset_id):
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

        Parameters
        ----------
        dataset_id : An unique identifier for the dataset
            
        variable_name : variable to be plotted
            
        quantize : To plot the quantize graph.
             (Default value = True)

        Returns
        -------
        Plot the variable distribution
        

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

        Parameters
        ----------
        dataset_id : An unique identifier for the dataset
            
        model_name : Model name
            
        model_type : Model type ['logistic', 'xgboost']
             (Default value = "xgboost")
        split_ratio : Train-Validation-Test ratio
             (Default value = "[0.7)
        0.3 :
            
        None]" :
            

        Returns
        -------
        Returns the model id
        

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

        return (response.json()['data']['progress'][0])



#### Model Details

    def model_summary(self,model_id):
        """

        Parameters
        ----------
        model_id : An unique identifier for the model
            

        Returns
        -------
        A dict containing performance metrics of the model

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

        Parameters
        ----------
        model_id : An unique identifier for the model
            

        Returns
        -------
        Plots the lift chart

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

        Parameters
        ----------
        model_id : An unique identifier for the model
            

        Returns
        -------
        Plot Receiver operating characteristic (ROC) curve
        

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

        Parameters
        ----------
        model_id : An unique identifier for the model
            

        Returns
        -------
        A dataframe containing feature importance
        

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

        Parameters
        ----------
        model_id : An unique identifier for the model
            
        split : 
             (Default value = 'train')

        Returns
        -------

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

        Parameters
        ----------
        dataset_id : An unique identifier for the dataset
            
        model_id : An unique identifier for the model
            
        threshold :
            

        Returns
        -------

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

        Parameters
        ----------
        df : A dataframe containing the user data, which should contains target variable, customer location details, and additional variables. 

        name : Name of the dataset
             (Default value = 'test')
        dv_col : Target variable column name
             (Default value = 'dv')
        dv_positive : Specify the good observation in the dataset 
             (Default value = '1')
        latitude : Latitutde column in the dataframe
             (Default value = "geo_latitude")
        longitude : Latitutde column in the dataframe
             (Default value = "geo_longitude")
        unique_col : Unique identifier in the dataset
             (Default value = 'geoiq_identifier_col')
        geocoding : If geocoding is required need to be true
             (Default value = 'false')
        address_col : Address column in the dataframe
             (Default value = '\'\'')
        pincode_col : Pincode column in the dataframe
             (Default value = '\'\'')
        model_id : An unique identifier for the model
        dataset_id : An unique identifier for the dataset
        user_selected_vars : Additional variables need to be passed in the creation of the model
             (Default value = '\'\'')

        Returns
        -------

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
        print(payload)
#         response = requests.request("POST", url, headers=headers, data=payload, files=files)

        os.remove(f'/tmp/{name}.csv')

        print(response.text)
        dataset_id = response.json()['data']['validation_dataset_id']

        return dataset_id


    ## LIFT Chart

    def create_validation_lift_chart(self,validation_dataset_id):
        """

        Parameters
        ----------
        validation_dataset_id : An unique identifier for the validation dataset


        Returns
        -------
        Plots the lift chart

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

        Parameters
        ----------
        validation_dataset_id : An unique identifier for the validation dataset


        Returns
        -------
        Plot Receiver operating characteristic (ROC) curve


        """


        ## AUC values
        url = "https://automlapis-us.geoiq.io/wrapper/prod/validationdatasets/v1.0/getvdevaluate"

        payload = json.dumps({
          "validation_dataset_id": validation_dataset_id
        })


        response = requests.request("GET", url, headers=self.headers, data=payload)

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

        Parameters
        ----------
        model_id : An unique identifier for the model

        split : train or validation
             (Default value = 'train')

        Returns
        -------

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

        Parameters
        ----------
        validation_dataset_id : An unique identifier for the validation dataset

        Returns
        -------

        """
        url = "https://automlapis-us.geoiq.io/wrapper/prod/validationdatasets/v1.0/getpsitable"

        payload = json.dumps({
          "validation_dataset_id": validation_dataset_id
        })

        response = requests.request("GET", url, headers=self.headers, data=payload)
        df_nested_list = pd.DataFrame(response.json()['data']['train']['data'])

        return df_nested_list

    
    
    def batch_prediction_download(self, validation_dataset_id):
        url = "https://automlapis-us.geoiq.io/wrapper/prod/validationdatasets/v1.0/getvdproperties"

        payload = json.dumps({
          "validation_dataset_id": validation_dataset_id
        })

        response = requests.request("POST", url, headers=self.headers, data=payload)
        return response.json()['data']['prediction_file']
    
    ## MODEL GET ALL DATASET


    def list_validation_datasets(self,model_id):
        """

        Parameters
        ----------
        model_id : An unique identifier for the model


        Returns
        -------
        Lists all the validationdatasets created by the user


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

        Parameters
        ----------
        model_id : An unique identifier for the model

        Returns
        -------

        """
        url = "https://automlapis-us.geoiq.io/wrapper/prod/model/v1.0/getdeployedmodelendpoint"

        payload = json.dumps({
        "model_id": model_id
        })


        response = requests.request("GET", url, headers=self.headers, data=payload)
        df_nested_list = pd.json_normalize(response.json()['data'])
        df_nested_list['created_at'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(df_nested_list['created_at']))

        return df_nested_list



    # Create Endpoint for Model / Deploy Model
    def create_deployed_model_endpoint(self,model_id):
        """

        Parameters
        ----------
        model_id : An unique identifier for the model

        Returns
        -------

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

        Parameters
        ----------
        latitude : latitude value
        longitude : longitude value
        model_id : An unique identifier for the model

        Returns
        -------
        Return a dataframe containing geoiq variables for new lat long selected in the model 

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

