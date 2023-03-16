![GeoIQ Automl package](https://geoiq.io/_next/image?url=%2Fimages%2Flogo.svg&w=256&q=75)

# GeoIQ-automl-python

[GeoIQ’s](https://geoiq.io/) ML python package makes it easy for data scientists and machine learning engineers to train your machine learning models at scale, evaluate your trained model, and use the model to make predictions and gain a better understanding of your data.

This package also helps geoiq data users to enrich the points (i.e. latitude/longitude pairs), census boundaries, and zipcodes in their data with geoiq features using Python.

Use this package to prepare datasets and build models as well as deploy, monitor and manage your models in production. 

## Getting started

1. **Installation** 

    Install via pip:

    ```bash
    pip install git+https://github.com/geoiq-io/geoiq_automl_us.git
    ```
    
    ```bash
    import geoiq_automl
    
    ## AutoML object
    result = geoiq_automl_us.automl('API KEY')
    ```
    
   
    
2. **Data Upload** <p align="left">
<img src="https://automl.geoiq.io/static/media/add-dataset.1bde6b66.svg" 
     width="80" 
     height="80"></p>

    User data upload:


    2.1 ****Without geocoding required****
    ```bash
    
    dataset_id = result.create_dataset(df, dataset_name ='test_dataset' ,dv_col = 'dv_90', dv_positive = '1',latitude_col = "geo_latitude" ,
          longitude_col = "geo_longitude",unique_col = 'geoiq_identifier_col',geocoding = 'F',
          address_col = '', pincode_col = '' , additional_vars = '[]')
    ```
    2.2 ****Geocoding required****
    ```bash
    
    dataset_id = result.create_dataset(df, dataset_name ='test_dataset' ,dv_col = 'dv_90', dv_positive = '1',latitude_col = '' ,
          longitude_col = '',unique_col = 'geoiq_identifier_col',geocoding = 'T',
          address_col = 'complete_address', pincode_col = 'pin_code_name' , additional_vars = '[]')
    ```

3. **Data Enrichment!**<p align="left">
<img src="https://cdn-icons-png.flaticon.com/512/300/300864.png" 
     width="60" 
     height="60"></p>

    The easiest way to enrich a file like this (with *all* the available GeoIQ features) is by running:

    ```bash
    result.data_enrichment(dataset_id,['p_retail_es_pt_500','mat_wall_mud_unbrnt_brck_perc_500','p_health_hp_np_1000'])
    ```

    S3 bucket link will be returned, in which you have your un-compressed GeoIQ data.
    
4. **Exploratory Data Analysis**<p align="left">
<img src="https://automl.geoiq.io/static/media/compare.4f516f22.svg" 
     width="80" 
     height="80"></p>

    Exploratory Data Analysis, or EDA, is an important step in any Data Analysis or Data Science project. We provide the detail investigation on the uploaded dataset to discover patterns, and anomalies (outliers), and form hypotheses based on our understanding of the dataset. 

    ```bash
    result.eda(dataset_id)
    ```
    ```bash
    ## Returned columns
    
    'column_name', 'column_type', 'iv', 'auc_1', 'auc_2', 'auc_3', 'bins', 'catchment', 'category', 'F_test_pvalue', 'T_test_pvalue', 'desc_name',
    'description', 'deviation', 'direction', 'id', 'ks', 'major_category', 'max_ks', 'mean', 'name', 'normalization_level',
    'normalization_level_id', 'roc', 'sd', 'sub_category_name', 'unique', 'unique_count', 'variable', 'vhm_hierarchy_id'
     ```
    
5. **Train a model**<p align="left">
<img src="https://automl.geoiq.io/static/media/train.6fddcfa4.svg" 
     width="80" 
     height="80"></p>
     
    This package helps you to create ML model on the top of your data, and provides the model results, which can be used to interpret the model according to your business need.

    ```bash
    result.create_custom_model( dataset_id=dataset_id,model_name="test_model",model_type = "xgboost", split_ratio ="[0.7,0.3,None]")
    ```

6. **Evaluate a model**<p align="left">
<img src="https://automl.geoiq.io/static/media/evaluate.dbfee37d.svg" 
     width="80" 
     height="80"></p>

   Evaluating a model is a core part of building an effective machine learning model. There are several evaluation metrics, like confusion matrix, cross-validation, AUC-ROC curve, etc. Different evaluation metrics are used for different kinds of problems.

    ```bash
    ## Model Summary
    result.model_summary(model_id)
    
    ## Confusion Matrix
    result.get_confusion_matrix(dataset_id,model_id,threshold)
    
    ## Lift Chart
    result.create_lift_chart(model_id)
    
    ## Roc curve
    result.create_roc_chart(model_id)
    
    
    ```
7. **Prediction from a model**<p align="left">
<img src="https://cdn-icons-png.flaticon.com/512/300/300834.png" 
     width="60" 
     height="60"></p>

     After you have created (trained) a model, you can deploy the model and request online (real-time) predictions. Online predictions accept one row of data and provide a predicted result based on your model for that data. You use online predictions when you need a prediction as input for your business logic flow.

    Before you can request an online prediction, you must deploy your model. Deployed models incur GeoIQ [credits](https://console.geoiq.io/in/credits). 
    


# Example


 





## See [demo_geoiq_automl.ipynb](https://github.com/geoiq-io/geoiq_automl_us/blob/main/tests/final_demo_geoiq_automl_us.ipynb) for more details.



## More resources

You can find our AutoML FAQ Page [here](https://geoiq.io/products/no-code-ml) and, access our Data Catalogue [here](https://catalog.geoiq.io/in).

    


## Contact us

For questions or issues with using this code, please [add a New Issue](https://github.com/geoiq-io/geoiq_automl/issues/new) and we'll respond as quickly as possible.

To get access to GeoIQ sample data please contact us [here](https://www.geoiq.io/contact)!

If you cannot find an answer to a question in here or at any of those links, please do not hesitate to reach out to GeoIQ Support (support@geoiq.io).
