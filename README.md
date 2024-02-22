# machine_learning_project-supervised-learning

## Project Outcomes
- Supervised Learning: use supervised learning techniques to build a machine learning model that can predict whether a patient has diabetes or not, based on certain diagnostic measurements.The project involves three main parts: exploratory data analysis, preprocessing and feature engineering, and training a machine learning model. 
### Duration:
Approximately 3 hours and 20 minutes.
### Project Description:
In this projects, you will apply supervised learning techniques to a real-world data set and use data visualization tools to communicate the insights gained from the analysis.

The data set for this project is the "Diabetes" dataset from the National Institute of Diabetes and Digestive and Kidney Diseases 
The project will involve the following tasks:

-	Exploratory data analysis and pre-processing: We will import and clean the data sets, analyze and visualize the relationships between the different variables, handle missing values and outliers, and perform feature engineering as needed.
-	Supervised learning: We will use the Diabetes dataset to build a machine learning model that can predict whether a patient has diabetes or not, using appropriate evaluation metrics such as accuracy, precision, recall, F1-score, and ROC-AUC. We will select at least two models, including one ensemble model, and compare their performance.

The ultimate goal of the project is to gain insights from the data sets and communicate these insights to stakeholders using appropriate visualizations and metrics to make informed decisions based on the business questions asked."

# Step 1 - A: EDA
I began by importing the necessary packages for the EDA process. I explored questions asked in the file.
The raw diabetes dataset has 768 rows and 9 columns. 8/9 are predictor columns, the 9th is the 'Outcome' column and is the target variable for this supervised learning project. The goal is to use the featutres to classify whetehr a patient has diabetes or not based on certain diagnoastic measurements. 
From just basic exploration, we find that there are no duplicated values and that there are no NULL/NA values.

Step 1 - A: Variable Relationships to the Outcome
A heat map was made to explore how the predictor variables are related to the outcome variable as well as how they are related to one another. Initially, I made a pair_plot to explore this but I quickly found out that there were too many variables to make an intelligent analysis using this mathod.
Every predictor variable appears to have a positive correlation with the 'Outcome' variable. This means that the higher a given patients' values in the predictor variables the more likely they are to be classified as having diabetes or not. 
From the annotated heatmap (or the correlation chart) we can see that of all the predictor variables, 'Glucose' has the strongest correlation (0.47) when determining if a patient has diabetes or not. Therefore, the top 3 predictor variables based on a heatmap using Pearson's corrleation coefficient are: 'Glucose' (0.47), 'BMI' (0.29), and 'Age' (0.24). Pregnancies are a close 4th with 0.22. 
Although none of the top 3 predictor variables showed an overwhelmingly strong relationship to the outcome variable alone (though 'Glucose' is nearly double the other features) they in combination paint a clearer picture of the type of patient that would be classified as having diabetes. It seems clear that this Dataset is looking at patients with Type-II diabetes because of the top 3 correlated predictor variables. Tyoe-II diabetes tends to afflict people who are older and overweight which is supported by the findings in the heat map. 
The weakest/least correlated predictor variables to the Outcome are 'Skin Thickness' (0.075) and 'Blood Pressure' (0.065). 

# B: Variable Relationships to Each Other:
## Moderate Correlations
Age and Pregnancies - According to the heatmap, 'Age' and 'Pregnancy' have the strongest (but moderate generally speaking) correlation (0.54) which is also a positive correlation. Something does feel a little strange about that relationship but it is also not surprising that the older a woman is the more pregnancies she may have. 
BMI and Skin Thickness - There is a moderate positive correlation (0.39). This appears accurate as BMI tends to indicate someones' weight and so more fat tissue may relate to thicker skin.
Skin Thickness and Insulin - positive moderate correlation (0.44)
Insulin and Glucose - positive weakish correlation which makes sense because insulin is required to regulate blood glucose levels.

### Weak Correlations
Diabetes Pedigree Function - has very week or near zero relationship to any of the other predictor variables as well as the outcome. Seemingly, this can be one of the features that is dropped when doing feature selection.
Pregnancies - Pregnancies has very weak or near-zero correlations with all other predictor variables except Age. Since Age and Outcome have a 0.24 coefficient and Pregnancy and Outcome have a 0.22 coefficient and the two are moderately strongly correlated with each other. Consider dropping Pregnancy or using pregnancy as another indicator because it seems like maybe we can reduce the dimensionality by dropping Pregnancy and keeping Age. 

# C: Exploring Outliers:
Pregnancy - Based on the box and whisker plot for Pregnancy, the obvious outliers for the column are any pregnancy count above 13. While even this number seems high, pregnancies do not necessarily equate to births so it may be an acceptable number. Based on the box plot, the pregnancy column has 14 values that are outliers. 

Blood Pressure - First of all, it appears as though the column BloodPressure is actually refering to BPM or even more specifically Resting Heart Rate because normally Blood Pressure is a value that is normally denoted as a fraction not an integer. This needs to be changed in the pre-processing stage. Secondly, for sure anyone with a BPM of 0 is impossible becasue those patients would be dead. From some basic research, I have found that a normal resting heart rate for adults is between 60 and 100 BPM but with some more atheltic people having a Resting HR of around 40 BPM. Therefore we will consider the values above and below the whiskers of the box plot to be outleirs. This means the values below 40 and above 105. There are 49 Outlier values in the Blood Presure column

Skin Thickness - Outliers will be considered values below the 50% mark (23) and above the top whisker (60). The number of outliers for this column are 379 Outliers in this column. There are only 768 values total. This means that half (or slightly more than half) of the SkinThickness values are outliers. Consider dropping the column.

Insulin - This is hard to tell which are outliers.  There are 377 values that are either 0 or Greater than 600. 

BMI - It is not biologically possible to have a BMI of 0 so those are automatically values that need to be adjusted. While it is rare for a BMI to be above 60, it is possible and since this is data from a diabetes dataset we should not discount the presence of extreme obesity as exsiting within the data. The number of patients with a BMI value of 0 is 11.

Age - No major outliers to be noted. No numbers seem out of place.

# D: Data Grouped by Diabetes Outcome 
Age - The average age of the patients is 33 years old.

Glucose -  The mean Glucose level for patients without diabetes is 109.98 which we'll call 110. The mean Glucose level for patients with diabetes is 141.24

BMI - The mean BMI of patients without diabetes is 30.3 and the BMI of patients with diabetes is 35.1

# E: Discussing Gender
There is no obvious gender column. That being said there is the pregnancy column. Any value with a pregnancy greater than 0 is obviously Female. The issue becomes whether or not we consider that Male are all those values that have a pregnancy value of 0. It is entirely possible that the entire dataset is only looking at women. This may be plausible since why else would there be a category for pregnancies, especially since the entire dataset is 768 records. Since only 111 patients have 0 pregnancies. This leads me to believe that the dataset is made up of only female patients.

# Step 2 - Preprocessing Data
## A: Dropping Unnecessary Columns and Column Adjustments
Pedigree Function - As has been seen in the heatmap, the DiabetesPedigreeFunction column is not strongly correlated to any other predictor variable and isn;t stringly correlated with the Outcome variable so we will be dropping it.

## B: Missing Values
I have a feeling that the dataframe, since there are many columns that have illogica '0' values in them, I think that is entirely possible that the dataset equated NaN with 0s. That being said I will not necessarily opperate on that assumption but I will replace the illogical '0' values

BMI - There were 11 records with a 0 in the BMI column. It is impossible to have a BMI of 0 so I have replace the '0' value with the mean which is 31.99.

BloodPressure  - there were 35 records where BloodPressure was 0. This person would be dead. I replaced those values with the mean which is around 69. The new min for the column is 24. This is also extremely low and likely needs to be adjusted. There are 9 records which have a BP of less than 45 which is extremely uncommon. Generally, any value lower than 50 is fairly uncommon but to account for women who might be younger or in better shape. I will replace the values les than 45 with 45. 
There are 2 patients with extremely high blood pressure. Since it is only 2/768 records, I will chalk them up do being acceptable outliers. 

Skin Thickness - There were a significant number of records with 0 in the SkinThickness column. It seems to me that it is impossible for a patient to have no skin thickness especially when the mean is around 26.6. So I replaced the zero values in Skin Thickness with the mean for the column. 
Also, since the IQR for that column ranges from 20.5 to 32, I am considering any value above 50 (to account for some variance) as an outlier. I am replacing those valuues that are above 50 with 50. There are 9 records to replace. 

Glucose - It seems impossible to have a Glucose level of 0. There are 5 records with the Glucose value of 0 that have been replaced by the column mean - 120.89. There are 117 records with Glucose levels above 155. This quite high but seeing as there are so many records and this is a diabetes dataset, I think it is acceptable to consider these as necessary/acceptable outliers.

Insulin - more than half of the patients have a zero as their insulin level. Even a healthy individual in a fasting state produces some insulin so I will replace the zero value withthe mean. There are some very high levels of insulin in this dataset but if these people have insulin resistance which can happen with type 2 diabetes, we can generally accept these high values as part of the study. 

## C: Scaling and Normalizing
Since the problem we are trying to solvce is a binary classification (ie Diabetic and Not Diabetic), I think it is important to scale the data despite the magnitudes of the data being relatively similar to one another. Additionally, it is important to normalize the data ahead of putting it into aML algorithm to make predicitions because the box and whisker plots of the columns have shown us that the data does not have a Gaussian distribution. In fairness, this dataset would not neccessarily need to be scaled but for the practice of getting used to doing it, I think its useful to engage with the tool. 
Made a copy of the dataframe a head of doing the transformations just in case.
Also, its important to note that the dataset is imbalanced. It is relatively small, only 768 records, but the split of diabetic to not diabetic is 500 (ND - not diabetic) to 268 (D - diabetic). This is essentially a 2:1 ratio. Since it is imbalanced, when splitting my data into training and testing I've activated the stratify parameter.
I used the StandardScaler to standardize the data. 

# Step 3 - Train the Models
I just to train the model using Logistic Regression as my simple classifier model since the task is binary classification. For the ensemble model I chose the Random Forest Classifier. Before any hyperparamter tuning or grid searching, I used several metrics to determin the best model. 

These were the best results:
Logistic Regression Accuracy: 0.7597402597402597
Logistic Regression Precision: 0.7073170731707317
Logistic Regression ROC AUC Score: 0.7085185185185187

Random Forest Classifier Accuracy: 0.7467532467532467
Random Forest Classifier Precision: 0.7027027027027027
Random Forest Classifier ROC AUC Score: 0.6857407407407407

The other scores were so low they were not worth sharing.
After doing the grid search, the best parameters for the Logistic Regression Model are: 
C = 0.01 and solver = newton-cg

The best parameters for the Random Forest Classifier Model are:
criterion= entropy, max_depth= 20, min_sample_split= 5




