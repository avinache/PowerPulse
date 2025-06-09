# PowerPulse
Machine learning model that can predict household energy consumption based on historical data

In the modern world, energy management is a critical issue for both households and energy providers. Predicting energy consumption accurately enables better planning, cost reduction, and optimization of resources. The goal of this project is to develop a **machine learning model** that can predict household energy consumption based on historical data. Using this model, consumers can gain insights into their usage patterns, while energy providers can forecast demand more effectively.
From this project we provide **actionable insights** into energy usage trends and deliver a **predictive model** that can help optimize energy consumption for households.

The initial step in our machine learning pipeline involved acquiring and preparing the dataset. The dataset was downloaded from the given source, extracted, and successfully imported into a Pandas DataFrame for further analysis and manipulation.

**Handling Missing and Inconsistent Data**
To ensure data quality, we began by inspecting the dataset for null, empty, or anomalous values. This included checking for NaNs and other placeholders that may indicate missing data. Where applicable, imputation techniques were applied to fill in missing values, using methods appropriate to the data type and distribution (e.g., mean or median imputation).

We also performed value counts on individual features to uncover uncommon, unusual, or inconsistent entries. These checks helped in detecting possible data entry errors or formatting issues that could distort model performance. Any inconsistent or unknown values were cleaned or corrected to maintain data integrity.

**Data Formatting and Structure**
Next, we standardized column names and formats to ensure uniformity throughout the dataset. This step involved correcting inconsistencies in naming conventions, aligning data types, and organizing the dataset for seamless analysis and model input.
Since the dataset consisted entirely of continuous numerical features, we confirmed that categorical encoding was not required, simplifying the preprocessing pipeline.

**Feature Selection and Correlation Analysis**
To focus on the most impactful variables, we conducted a correlation analysis. This helped identify and retain features that had strong relationships with the target variable, while filtering out those with low or negligible predictive power.

**Outlier Detection and Treatment**
We performed outlier analysis using statistical methods to identify extreme values that could skew model training. These outliers were either removed or adjusted using techniques such as winsorization to maintain the integrity of the feature distributions.

**Feature Scaling**
Finally, we applied both normalization and standardization techniques to rescale the selected numerical features. This step ensured that all input variables were on a similar scale, which is particularly important for algorithms sensitive to feature magnitude, such as linear models.

**Feature and Target Selection**
To develop an effective prediction model for household energy consumption, we began by selecting relevant input features and defining the target variable. After examining the dataset, we excluded the 'Date' column due to its non-numeric, non-predictive nature, and also removed 'Active Power' from the feature set, as it was designated as the target variable for prediction.
These continuous numerical variables were considered potentially influential in predicting active power consumption.

**Target Variable Definition**
The target variable, Active Power, represents the real power consumed by a household and is measured as a continuous variable. Its numeric and unbounded nature made it suitable for regression-based modeling approaches.

**Model Selection**
Given the continuous nature of the target and the linear relationships observed during correlation analysis, we selected the Linear Regression algorithm as a baseline model. This approach is not only interpretable but also efficient for establishing a foundational performance benchmark.

**Data Splitting**
To evaluate the model’s generalizability, the dataset was split into training and testing subsets. This allows the model to learn patterns from the training data and validate its performance on previously unseen data. A standard train-test split ratio (e.g., 80/20 or 70/30) was used to ensure a fair evaluation.

**Model Training and Prediction**
The **Linear Regression, KNN, Decision Tree** model was trained using the selected input features. After training, the model was used to generate predictions on both the training and test sets. This step helped assess whether the model was underfitting or overfitting the data and also help to find the suitable model for the given data set.

**Performance Evaluation**
To quantify the model's performance, we used the Root Mean Squared Error (RMSE) as the primary evaluation metric. RMSE provides a clear measure of the average magnitude of prediction errors, penalizing larger deviations more heavily. Lower RMSE values indicate better alignment between predicted and actual consumption values.
