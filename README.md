# Target Audience for Direct Marketing in Starbucks Rewards Mobile App

# Requirements
seaborn==0.9.0

numpy==1.16.4

pandas==0.25.0

matplotlib==3.1.0

scikit_learn==0.21.3

# Motivation
This is my capstone project for Udacity Data Science Nanodegree. 

In this project, I analyze the customer behavior in the Starbucks rewards mobile app. After signing up for the app, customers receive promotions every few days. The aim of this project is to identify target audience for a successful marketing campaign in direct marketing. Which customers love coupons? Which don't? What types of offers send to whom?

To solve this problem, I performed **customer segmentation using K-means clustering technique**. 
The idea is to divide app users into major groups - those more prone to discounts vs. those more keen on bogos vs. those that are not interested in promotions at all.

The full technical report can be found on my blog - [https://www.cross-validated.com/Starbucks-Rewards-Program/](https://www.cross-validated.com/Starbucks-Rewards-Program/)

# Files
```
Starbucks_rewards
    |-- data
        |-- portfolio.json
        |-- profile.json
        |-- starbucks_customer_level.csv
        |-- starbucks_offer_level.csv
        |-- transcript.json
       
    |-- 1_Starbucks_cleaning.ipynb
    |-- 2_Starbucks_EDA.ipynb
    |-- 3_Starbucks_modeling.ipynb
    |-- 4_Starbucks_refinement.ipynb
    
    |-- README.md
	  |-- requirements.txt
	  |-- starbucks_rewards.py	
```

Data folder contains all initial datasets in json format (portfolio, profile, transcript) and two csv files generated after performing cleaning and aggregation steps.

The modeling was performed in Jupyter Notebooks and refactored in `starbucks_rewards.py`
 - `1_Starbucks_cleaning.ipynb` - cleaning, merging, aggregating steps
 - `2_Starbucks_EDA.ipynb` - explorative data analysis
 - `3_Starbucks_modeling.ipynb` - modeling part: imputing, one-hot encoding, scaling, clustering, results evaluation
 - `4_Starbucks_refinement.ipynb` - all the above steps refactored as functions. Modeling under different analytical assumptions


# Results


# Acknowledgements

The dataset was provided by Starbucks within [Udacity Data Science Nanodegree](https://www.udacity.com/course/data-scientist-nanodegree--nd025)

