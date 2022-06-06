# Austin Crime Project

by: Alejandro Garcia, Matthew Luna, Kristofer Rivera, Oliver Ton
<br>
Date: June 2022

This repository contains all files, and ipython notebooks, used in the Austin Crime Data Science Capstone Project.


___

## Table of Contents

- I. [Project Summary](#i-project-summary)<br>
- II. [Project Planning](#ii-project-planning)<br>
    - [1. Project Goals](#ii-project-goals)<br>
    - [2. Business Goals](#iii-gusiness-goals)<br>
    - [3. Project Description](#i-project-description)<br>
- III. [Data Dictionary](#iii-data-dictionary)<br>
- IV. [Outline of Project Plan](#iv-outline-of-project-plan)<br>
    - [1. Project Plan](#1-plan)<br>
    - [2. Data Acquisition](#2-acquire)<br>
    - [3. Data Preparation](#3-prepare)<br>
    - [4. Data Exploration](#4-explore)<br>
    - [5. Modeling & Evaluation](#5-model)<br>
    - [6. Product Delivery](#6-deliver)<br>
- V. [Conclusion](#v-conclusion)<br>
- VI. [Instructions For Recreating This Project](#vi-instructions-for-recreating-this-project)<br>

___

## I. Project Summary

<details><summary><i>Click to expand</i></summary>

This project analyzes crime data from the city of Austin for the years 2018 through 2021. The goal of the project was to develop a deeper understanding of the factors that drive crime in Austin, TX and the indicators of whether or not a particular case will be solved/cleared. Using exploratory visualizations and statistical analysis we investigated several indicators of clearance such as crime type, location, seasonality, and timeliness of the report. Ultimately, we found all of these to be important indicators of clearance and built a predictive classification model using a Naive Bayes algorthim that can predict clearance status on unseen data with an accuracy of 89% and ROC-AUC Score of .81. This accuracy outperforms the baseline by ~ 11%.

</details>

___

## II. Project Planning

<details><summary><i>Click to expand</i></summary>

### Project Goals

Identify key indicators for successfully closing a crime case for the city of Austin given data for the years 2018 - 2021.

### Problem Statement

What factors contribute to whether or not a crime is solved/closed in the city of Austin?

### Project Description

- This project will dive into crime data from the city of Austin for the years 2018 through 2021. 
- Having a deeper understanding of the crime in Austin will allow for improved public safety 
outcomes. 
- This project will cover key indicators for successfully closing a case, the most frequent 
types of crimes, Austin city district crime rate, and the seasonality of crimes. 
- Our goal is that this project will guide the allocation of resources toward improving public 
safety. Let’s keep Austin weird! And safe.

### Initial Hypotheses

- We predict that there is a relationship between the type of crime and clearance status.
- We predict there is a relationship between city council district and clearance status.
- We predict that there is a relationship between higher seasonal levels of crime and clearance status.
- We predict that the difference in time between when an incident occurred and when it was reported relates to the clearance status of the case.


</details>

___

## III. Data Dictionary

<details><summary><i>Click to expand</i></summary>

| Name                        | Definition    | API Field Name | Data Type       
| :-----                      | :-----        | :-----         | :-----
| Incident Number             | Incident report number | incident_report_number | Number
| Highest Offense Description	| Description | crime_type | Plain Text
| Highest Offense Code        | Code        | ucr_code | Number
| Family Violence             | Incident involves family violence? Y = yes, N = no | family_violence | Plain Text
| Occurred Date Time          | Date and time (combined) incident occurred | occ_date_time | Date & Time
| Occurred Date	              | Date the incident occurred | occ_date | Date & Time
| Occurred Time	              | Time the incident occurred | occ_time | Number
| Report Date Time	          | Date and time (combined) incident was reported | rep_date_time | Date & Time
| Report Date	                | Date the incident was reported |rep_time | Date & Time
| Report Time	                | Time the incident was reported | location_type | Number
| Location Type	              | General description of the premise where the incident occurred | location_type | Plain Text
| Address	                    | Incident location | address | Plain Text
| Zip Code	                  | Zip code where incident occurred | zip_code | Number
| Council District	          | Austin city council district where incident occurred | council_district | Number
| APD Sector	                | APD sector where incident occurred | sector | Plain Text  
| APD District	              | APD district where incident occurred | district | Plain Text
| PRA	                        | APD police reporting area where incident occurred | pra | Plain Text
| Census Tract	              | Census tract where incident occurred | census_tract | Number
| Clearance Status	          | How/whether crime was solved (see lookup) | clearance_status | Plain Text
| Clearance Date	            | Date crime was solved | clearance_date | Date & Time
| UCR Category	              | Code for the most serious crimes identified by the FBI as part of its Uniform Crime Reporting program | ucr_category | Plain Text
| Category Description	      | Description for the most serious crimes identified by the FBI as part of its Uniform Crime Reporting program | category_description | Plain Text
| X-coordinate	              | X-coordinate where the incident occurred | x_coordinate | Number
| Y-coordinate	              | Y-coordinate where incident occurred | y_coordinate | Number
| Latitude	                  | Latitude where incident occurred | latitude | Number
| Longitude	                  | Longitude where the incident occurred | longtitude | Number
| Location	                  | 3rd party generated spatial column (not from source) | location | Location

 

Additionally, a set of features were added to the data set:

 
| Name                  | Definition    | Data Type                                   
|:-----                 | :-----        |:-------------------------                  
| geometry              | A list of coordinates | Multi-Polygon and Polygon
| time_to_report        | The difference in time between when a crime occurred and when it was reported. | Time

</details>

___

## IV. Outline of Project Plan

The overall process followed in this project is as follows: 

Plan  -->  Acquire   --> Prepare  --> Explore  --> Model  --> Deliver

---
### Data Acquisition

<details><summary><i>Click to expand</i></summary>


**Acquisition Files:**
- acquire.ipynb: Contains all the steps and decisions taken in the data acquisition phase of the pipeline.
- acquire.py: Contains functions used for acquiring the Austin crime data using an API or reading the data from a .csv file.

**Steps Taken:**

- The data was gathered from publicly available data provided by the Austin Police Department on data.austintexas.gov.
- We created a function to automate gathering the data from the provided API and caching it locally as a CSV file. 
- Our initial data set included 500,000 rows and 31 columns. 
- For ease of use and relevancy, we decided to limit our data to crimes reported between the years 2018 and 2021. 
- After removing data outside this time frame, we were left with 401,955 rows.

**Additional Steps:**
- For visualizing geospatial data download the shapefile for boundaries zipcode tabulation areas at this [website](https://data.austintexas.gov/dataset/Boundaries-Zip-Code-Tabulation-Areas-2017/nf4y-c7ue)
- Merge the dataframes and then create a new csv file.

</details>

### Data Preparation

<details><summary><i>Click to expand</i></summary>git

**Preparation Files:**
- prepare.ipynb: Contains all steps and decisions made in the data preparation phase of the pipeline.
- prepare.py: Contains functions used for preparing the data for exploration and modeling. Also contains used for univariate exploration in the prepare notebook.

**Steps Taken:**

- After investigating columns with missing values, we decided to drop 15 columns entirely that we deemed to be unuseful or redundant. 
- Next, we made decisions on how to handle the missing values in our remaining 16 columns. 
- For 7 columns, including clearance_status, clearance_date, zip_code, sector, district, latitude, and longitude, we decided that we could not reasonably impute nulls with a value and dropped all missing rows. 
- We had 753 missing values for location_type values which we decided to add to the Other / Unknown value. 
- We had 1438 missing values for council_district which we decided to impute as the most common district. 
- For readability we renamed a few columns.
- The target variable (clearance_status) originally has the values N, O, and C which are not very meaningful. These were changed to the more human readable values not cleared, cleared by exception, and cleared by arrest.
- We cast the columns to more appropriate data types where necessary.
- We combined the "cleared by arrest" and "cleared by exception" values into one "cleared" value for ease of use.

</details>

### Exploratory Analysis

<details><summary><i>Click to expand</i></summary>

**Exploratory Analysis Files:**
- explore.py: Contains all functions used in the exploration phase of the pipeline and all functions used for producing visualizations in the final notebook.
- univariate_analysis.ipynb: Contains steps and takeaways from the univariate analysis of the data.
- rivera_explore.ipynb: Contains steps taken in answering the question, which types/categories of crime are not getting solved?
- garcia_explore.ipynb: Contains steps taken in answering the question, does the clearance status of a case depend on the amount of time between when a crime occurred and when it was reported.
- oliver_notebook.ipynb: Contains steps taken in answering the question, is there seasonality in crime?
- matt_explore.ipynb: Contains steps taken in answering the question, are there certain city council districts with disproportiate levels of crime?

**Steps Taken:**
- We began exploring the data by investigating the distributions of values in the various features contained in the data.
- Next, we split the data into three sets: train, validate, and test. Only the train dataset is explored from this point on.
- The relationship between types of crime and clearance status was investigated.
- The relationsihp between the time to report a crime and clearance status was investigated.
- The seasonality of the data was investigated.
- The relationship between council district and clearance status was investigated.

</details>

### Modeling

<details><summary><i>Click to expand</i></summary>

**Modeling Files:**
- model.ipynb: Contains all steps and decisions made in the modeling phase.
- model.py: Contains functions and objects used for building machine learning models.
- evaluate.py: Contains functions used for evaluating model performance.

**Steps Taken:**
- We decided to use roc auc score and accuracy as our metrics for measuring model performance.
- A baseline model was established to serve as simple model to compare model performance to.
- Several machine learning algorithms were used, provided by sklearn, with mostly default values to determine which algorithm provides the best performance for making predictions on the train dataset. The top performing models were evaluated on the validate set and top performer from those was chosen to move forward with.
- For the top performing model the hyper-parameters were modified to determine which set of hyper-parameters can provide the best performance on the validate set. The top performing model from these was chosen to move forward with.
- The top performing model was evaluated on the test dataset to determine how it could be expected to perform on unseen data.

</details>

___

## V. Conclusion

<details><summary><i>Click to expand</i></summary>

Our exploratory data analysis provided several key insights surrounding the factors that drive crime and whether or not a case gets cleared. We identified that several of the top crimes in terms of frequency are also the lowest in terms of clearance rate. We identified disproportionate levels of crime in certain council districts, clear seasonal trends, and the importance of timely reporting in ensuring a case gets cleared. Our best performing model was a Naive Bayes model that can predict case clearance with 89% accuraccy and a ROC-AUC score of .81 significant outperforming the baseline. We hope our insights and predictive model can be used to guide policy making and allocation of resources towards improving public saftery outcomes in Austin, TX. 

</details>

___

## VI. Instructions For Recreating This Project

<details><summary><i>Click to expand</i></summary>

1. Clone this repository into your local machine by running the following command in a terminal:
```bash
git clone git@github.com:austin-crime/austin-crime.git
```
2. You will need Pandas, Numpy, Matplotlib, Seaborn, and SKLearn installed on your machine.
3. Additionally you will need to install the following packages:
    - [Sodapy](https://github.com/xmunoz/sodapy)
    - [Geopandas](https://geopandas.org/en/stable/)
These can be installed by running the following commands in a terminal:
```bash
pip install sodapy
pip install geopandas
```
4. (Optional) Creating an app token is generally recommended for using the Socrata API with sodapy, however for the purposes of recreating this project it is not necessary. If you are interested in creating an app token follow the instructions [here](https://support.socrata.com/hc/en-us/articles/210138558-Generating-an-App-Token). Put your app token in a env.py file like so:
```python
app_token = 'your_app_token'
```
5. Now you can start a Jupyter Notebook session (or your favorite iPython notebook environment) and execute the Final_Report.ipynb notebook.

</details>

[Back to top](#austin-crime-project)