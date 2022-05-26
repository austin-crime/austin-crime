# Austin Crime Project

by: Alejandro Garcia, Matthew Luna, Kristofer Rivera, Oliver Ton    June 2022

This repository contains all files, and ipython notebooks, used in the Austin Crime Data Science Capstone Project.


___

## Table of Contents

* I. [Project Summary](#i-project-summary)<br>
* II. [Project Planning](#ii-project-planning)
    [1. Project Goals](#ii-project-goals)<br>
    [2. Business Goals](#iii-gusiness-goals)<br>
    [3. Project Description](#i-project-description)<br>
* III. [Data Dictionary](#iii-data-dictionary)<br>
* IV. [Outline of Project Plan](#iv-outline-of-project-plan)<br>
    [1. Project Plan](#1-plan)<br>
    [2. Data Acquisition](#2-acquire)<br>
    [3. Data Preparation](#3-prepare)<br>
    [4. Data Exploration](#4-explore)<br>
    [5. Modeling & Evaluation](#5-model)<br>
    [6. Product Delivery](#6-deliver)<br>
* V. [Conclusion](#v-conclusion)<br>
* VI. [Instructions For Recreating This Project](#vi-instructions-for-recreating-this-project)<br>

___

## I. Project Summary

___

## II. Project Planning

<details><summary><i>Click to expand</i></summary>

### Project Goals

- Create a machine learning model that can predict whether or not a crime is solved/closed in Austin.

### Problem Statement

- What factors contribute to whether or not a crime is solved/closed in the city of Austin?

### Project Description

- This project will dive into crime data from the city of Austin for the years 2018 through 2021. 
- Having a deeper understanding of the crime in Austin will allow for improved public safety 
outcomes. 
- This project will cover key indicators for successfully closing a case, the most frequent 
types of crimes, Austin city district crime rate, and the seasonality of crimes. 
- Our goal is that this project will guide the allocation of resources toward improving public 
safety. Let’s keep Austin weird! And safe.

### Initial Questions

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

 

| Name                  |Datatype      | Definition                                             | Possible Values    |
|:-----                 | :-----       |:------------------------------                         |:-----              |


</details>

___

## IV. Outline of Project Plan

The overall process followed in this project is as follows:

[Trello Board](https://trello.com/b/XCIqIEMJ/austin-crime-capstone)
 

###  Plan  -->  Acquire   --> Prepare  --> Explore  --> Model  --> Deliver

=======
<br>
Plan  -->  Acquire   --> Prepare  --> Explore  --> Model  --> Deliver

---
### Data Acquisition

<details><summary><i>Click to expand</i></summary>


**Acquisition Files:**


**Steps Taken:**

- The data set was gathered from publicly available data provided by the Austin Police 
Department on data.austintexas.gov.
- We created a function to automate gathering the data from the provided API and caching it 
locally as a CSV file. 
- Our initial data set included 500,000 rows and 31 columns. 
- For ease of use and relevancy, we decided to limit our data to crimes reported between the 
years 2018 and 2021. 
- After removing data outside this time frame, we were left with 401,955 rows. 

</details>

### Data Preparation

<details><summary><i>Click to expand</i></summary>git

**Preparation Files:**


**Steps Taken:**

- After investigating columns with missing values, we decided to drop 15 columns entirely that we deemed to be unuseful or redundant. 
- Next, we made decisions on how to handle the missing values in our remaining 16 columns. 
- For 7 columns, including clearance_status, clearance_date, zip_code, sector, district, latitude, and longitude, we decided that we could not reasonably impute nulls with a value and dropped all missing rows. 
- We had 753 missing values for location_type values which we decided to add to the Other / Unknown value. 
- We had 1438 missing values for council_district which we decided to impute as the most common district. 

</details>

### Exploratory Analysis

<details><summary><i>Click to expand</i></summary>

**Exploratory Analysis Files:**


**Steps Taken:**


</details>

### Modeling

<details><summary><i>Click to expand</i></summary>

**Modeling Files:**


**Steps Taken:**


</details>

___

## V. Conclusion

<details><summary><i>Click to expand</i></summary>



</details>

___

## VI. Instructions For Recreating This Project

<details><summary><i>Click to expand</i></summary>



</details>

[Back to top](#austin-crime-project)