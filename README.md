Predicting Breast Cancer in a patient

Abstract: 

--> Breast cancer represents one of the diseases that make a high number of deaths every year.
-->It is the most common type of all cancers and the main cause of women's deaths worldwide.
--> Classification and data mining methods are an effective way to classify data.
--> Especially in the medical field, where those methods are widely used in diagnosis and analysis to make decisions.

Problem Statement:

--> Given the details of cell nuclei taken from breast mass, predict whether or not a patient has breast cancer using the Ensembling Techniques.
--> Perform necessary exploratory data analysis before building the model and evaluate the model based on performance metrics other than model accuracy.

###Dataset Information:
The dataset consists of several predictor variables and one target variable, Diagnosis.
The target variable has values 'Benign' and 'Malignant', where 'Benign' means that the cells are not harmful or there is no cancer and 'Malignant' means that the patient has cancer and the cells have a harmful effect

###Variable Description:

###Column Description

--> radius Mean of distances from center to points on the perimeter
--> texture Standard deviation of gray-scale values
--> perimeter Observed perimeter of the lump
--> area Observed area of lump
--> smoothness Local variation in radius lengths
--> compactness perimeter^2 / area - 1.0
--> concavity Severity of concave portions of the contour 
--> concave points number of concave portions of the contour
--> symmetry Lump symmetry
--> fractal dimension "coastline approximation" - 1
--> Diagnosis Whether the patient has cancer or not? ('Malignant','Benign')


The mean, standard error and "worst" or largest (mean of the three largest values) of these features were computed for each image, resulting in 30 features.
For instance, field 3 is Mean Radius, field 13 is Radius SE, field 23 is Worst Radius.

###Scope:

● Analysing the available data and exploring relationships among given variables

● Data Pre-processing

● Training SVM classifier to predict whether the patient has cancer or not

● Assess the correctness in classifying data with respect to efficiency and effectiveness of the SVM classifier in terms of accuracy, precision sensitivity, specificity and AUC ROC

● Tuning the hyperparameters of SVM Classifier provided by the scikit-learn library

#**Learning Outcome:**
The students will get a better understanding of how the variables are linked to each other and build an SVM model. Apart from various performance measures, they will also learn about hyperparameter tuning with cross-validation to improve these scores.



# BREAST CANCER

--> Breast cancer is cancer that develops from breast tissue.

--> It's a disease in which malignant(cancer) cells from in the tissues of the breast.

--> When a tumor is diagnosed as benign, doctors will usually leave it alone rather than remove it.


# Technology 

##Machine learning with python

### -->  Python libraries such as pandas, numpy, Sklearn library used with manu subordinate models such as model_selection, classifiers, and metrics etc...

### --> Google Colab Notebook for implementation purpose.

--> ![MLA_Image](https://user-images.githubusercontent.com/110006271/205318951-c1f714d8-ff3d-4abb-8f56-95464686664f.jpg)

# --> Dataset

## --> Attributes:

### --> Diagnosis: The diagnosis of breast tissues(1=malignant, 0=benign)

### --> mean_radius: mean of distances from center to points on the perimeter

### --> mean_texture: standard deviation of gray-scale values.

## --> mean_perometer: mean size of the core tumor.

## --> mean_area

### --> mean_smoothness: mean of local variation in radius lengths.

# Exploring the DATA

# Data Preprocessing

## Splitting X, y for the train and test data.

## Applying the regression/ classification models on the dataset..

# Classification and Regression Tree:

## --> It's the classification techniques used machine learning where the data is continously split according to a certain parameter.

# Support Vector Machine (SVM):

## SVM classifier is a supervised machine learning models with associated learning alogorithms that analyze data used for classification and regression analysis.

# GaussianNB:

# K Nearest Neighbors (KNN)


# Model used vs Accuracy -- Performance comparions:

![Performance comparision](https://user-images.githubusercontent.com/110006271/202886344-7b63b2c9-ef28-49c3-ad62-66f991aa6f0d.png)


Scaled performance comparions


![Scaled performance comparions](https://user-images.githubusercontent.com/110006271/202886372-4422f0cb-19e0-49ad-ade7-5488e3e9db45.png)


Displaying the confision matrix

![Displaying the confusion matrix](https://user-images.githubusercontent.com/110006271/202886488-2ea953e7-550d-494a-a3f4-2a947482d649.png)


# Classification report:

## --> A cassification report is used to measure the quality of predictions from a classification algrithm.

    --> The report shows the main classification metrics precision, recall and f1-score on a per-class basis.
    
    --> The metrics are calculated by using true and false positives, true and false negatives. 
    
    --> Positive and negative in this case are generic names for the preidcted classed.
    
    --> There are four ways to chech if the predictions are right or wrong.
    
        --> TP/True Positive: When a case was positive and predicted positive.
        
        --> TN/True Negative: When a case was negative and predicted negative.
        
        --> FN/False Negative: When a case was positive and predicted negative.
        
        --> FP/False Positive: When a case was Negative and predicted positive.
        
    Precision: Accuracy of positive predictions. precision = TP/(TP+FP)
    
    Recall: Fraction of positives that were correctly identified. Recall = TP/(TP/ FN)
    
    F1 Score: Harmonic mean of precision and recall F1 Score = 2*(recall*precision)/(recall+precision)
    
    
Conclusion:

    --> Based on accuracy, confusion matrix and classification report. We can say that Support Vector Classifiter is best for our problem.
    
