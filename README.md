# Austin Crime Project

## This repository contains the work for the Codeup Data Science Capstone Project

## Alejandro Garcia, Matthew Luna, Kristofer Rivera, Oliver Ton    June 2022

Table of Contents
---
 
* I. [Project Description](#i-project-description)<br>
* II. [Project Goals](#ii-project-goals)<br>
* III. [Business Goals](#iii-gusiness-goals)<br>
* IV. [Data Dictionary](#iv-data-dictionary)<br>
* V. [Data Science Pipeline](#v-project-planning)<br>
[1. Project Plan](#1-plan)<br>
[2. Data Acquisition](#2-acquire)<br>
[3. Data Preparation](#3-prepare)<br>
[4. Data Exploration](#4-explore)<br>
[5. Modeling & Evaluation](#5-model)<br>
[6. Product Delivery](#6-deliver)<br>
* V. [Project Reproduction](#vi-to-recreate)<br>
* VI. [Key Takeaway](#vii-takeaways)<br>
* VI. [Next Steps](#viii-next-steps)<br>

## I. Project Description

-------------------



 

## II. Project Goals

-------------


1. Create scripts to perform the following:

                a. acquisition of data from GitHub's website

                b. preparation of data

                c. exploration of data

                d. modeling

2. Build and evaluate _______ models to predict ______
 

## III. Business Goals

--------------



 

## IV. Data Dictionary

---------------


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

 


## V. Project Planning

----------------

The overall process followed in this project is as follows:
[Trello Board](https://trello.com/b/XCIqIEMJ/austin-crime-capstone)
 

###  Plan  -->  Acquire   --> Prepare  --> Explore  --> Model  --> Deliver


--------------


### 1. Plan




### 2. Acquire



### 3. Prepare

This functionality is stored in the python script "prepare.py". It will perform the following actions:


- split the data into 3 datasets - train/test/validate - used in modeling

  - Train: 56% of the data

  - Validate: 24% of the data

  - Test: 20% of the data

 

### 4. Explore

Answer the following questions using data visualization and statistical testing:



### 5. Model


Compare the models against the baseline and each other based on the accuracy score from the validate sample. We sorted by ascending dropoff in accuracy from train to validate to guard against choosing an overfit model. 

Test the best performing model on witheld test data.


### 6. Deliver


### VI. To recreate

----------------



 

### VII. Takeaways

-------------


 

### VIII. Next Steps

-------------

