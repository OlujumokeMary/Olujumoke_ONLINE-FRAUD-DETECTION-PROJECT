# ONLINE-FRAUD-DETECTION - CASE STUDY FOR BLOSSOM BANK

Blossom Bank also known as BB PLC is a multinational financial services group, that offers retail and investment banking, pension management, asset management and payments services, headquartered in London, UK. Blossom Bank wants to build a Machine Learning model to predict online payment fraud.

# Methodology:

The objective of this project is to develop a Machine Learning model that can forecast instances of online payment fraud for Blossom Bank. To achieve this goal, we need to construct a Machine Learning model capable of detecting fraudulent transactions conducted online.
To begin, we must import essential libraries for data manipulation and visualization, including pandas (as pd), numpy (as np), matplotlib.pyplot (as plt), and seaborn (as sns). 
Next, we need to load the dataset by executing the command "data=pd.read_csv("")" from our local computer. Once the data is loaded, we can examine its first five columns and the last five columns using the "data.head()" function.

# Conducting Exploratory Data Analysis (EDA)

To gain insights into the dataset and inform our model building, I need to perform exploratory data analysis. I started by visualizing the relationships between the target variable and some critical features using the seaborn library's "countplot" function. For instance, I used "sns.countplot(x='isFraud', data=data)" to plot the count of fraud and non-fraud transactions.

Next, I explore the correlations between the variables by creating a heatmap. This visualization method allows to examine the strength and direction of the relationships between pairs of variables.

For univariate analysis, I used the "countplot" function to display the distribution of the target variable. This visualization method helps to understand the proportion of fraud and non-fraud transactions in the dataset.

Finally, for multivariate analysis, I utilize the "boxplot" and "pairplot" functions to examine the relationships between some of the variables. A boxplot can reveal the distribution and outliers of a particular variable, while a pairplot can depict pairwise relationships between multiple variables in a single figure.

# Perform Feature Engineering

Separate the target variable from the dataset by droping the target varibale before encoding (in the case, isFraud was dropped)
Encode categorical variables - This involves converting of categorical variables/columns into numerical variables for the purpose of building the machine learning model.

# Selection, Training, and Validation of Models

To develop and evaluate the machine learning models,I split the dataset into training and testing sets.I accomplished this by importing "train_test_split" from the "sklearn.model_selection" module.

After splitting the data, I proceed to train and test four supervised learning models. Specifically, I import and train four models: LogisticRegression from "sklearn.linear_model," DecisionTreeClassifier from "sklearn.tree," KNeighborsClassifier from "sklearn.neighbors," and RandomForestClassifier from "sklearn.ensemble."

To assess the performance of the models, I import the "Classification report" from "sklearn" to view the model result of each of the four supervised learning models used in this machine learning project. This report can help evaluate the precision, recall, and F1-score of each model.

Furthermore, I import the "Confusion Matrix" to analyze the model results. This matrix can help assess the true positives, false positives, true negatives, and false negatives of each model and gain further insights into its performance.

# Interpretation of Model Results:

The classification report provides a summary of the performance evaluation metrics of four distinct classification models: Logistic Regression, Decision Tree Classifier, K-Nearest Neighbors (KNN) Classifier, and Random Forest Classifier (RF).

Each report displays the precision, recall, and F1-score for each class, as well as the support and accuracy for the model. Precision is the ratio of correctly predicted positive observations to the total predicted positive observations, while recall is the ratio of correctly predicted positive observations to the total actual positive observations. F1-score is the harmonic mean of precision and recall.

The support indicates the number of observations in each class, while accuracy represents the overall percentage of accurately classified observations.

In particular, precision is the proportion of correctly predicted positive observations (true positives) to the total predicted positive observations (true positives and false positives). Recall, on the other hand, is the ratio of correctly predicted positive observations (true positives) to the total actual positive observations (true positives and false negatives). The F1-score is the weighted average of precision and recall.

Accuracy, as stated, is the percentage of observations that were correctly classified by the model. It is calculated by dividing the sum of true positives and true negatives by the total number of observations in the dataset.
Notably, TN refers to true negatives, TP denotes true positives, FN represents false negatives, and FP indicates false positives.

From my result, the Precision rate for Logistic Regression is 55% and its Recall rate is 35%, resulting in an F1-score of 43%. However, due to the relatively low Recall rate, Logistic Regression may not be as effective in detecting fraudulent transactions compared to other classification models. The K-Nearest Neighbors (KNN) Classifier achieved a precision rate of 78% and a recall rate of 50%, resulting in an F1-score of 61%. However, its relatively low recall rate indicates that it may not be the best choice for detecting fraudulent transactions when compared to other classifiers. The Random Forest Classifier (RF) is a reliable tool for identifying fraudulent transactions due to its impressive Recall rate of 79%. Nonetheless, the model's high Precision rate of 97% could potentially cause inconvenience for customers during their transactions. Decision Tree Classifier - Based on the results, the Decision Tree Classifier appears to be the most effective out of the four classifiers that were utilized. This particular model achieved a high Recall rate of 81% for identifying fraudulent transactions, which was the primary objective of training and testing the models.

# What Matrics are the most important

The most crucial matrics in this case is the Recall matrics as it can detect fraudulent transactions in the dataset. A higher recall value suggests that the model has a lower number of false negatives, meaning the model can better identify all the positive instances. As a result, the business should be more concerned about false negatives.

# Business benefit:
Reducing fraud losses and increasing customer trust.
