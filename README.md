# Bike Sharing Demand Prediction
This project predicts bike-sharing demand using historical data. The problem is tackled using both regression and classification models.

The project also includes hyperparameter tuning for performance optimization and a deployment-ready model.
## Problem Description
- **Regression Task**: Predict the total number of bike rentals (`cnt`) based on weather and time-related features.
- **Classification Task**: Predict whether bike rentals exceed 500 in an hour (high demand).

Accurate predictions can help city planners optimize bike-sharing systems and manage resource allocation.
## Dataset
The dataset contains hourly data for bike rentals. Key features include:
- **Weather Features**: `temp`, `atemp`, `humidity`, `windspeed`.
- **Time Features**: `hour`, `weekday`, `month`, `season`.
- **Target Variables**:
  - `cnt`: Total bike rentals (regression).
  - `high_demand`: Binary target for classification (`1` if `cnt > 500`, else `0`).

Source: UCI Machine Learning Repository - Bike Sharing Dataset.

## Exploratory Data Analysis and Feature Engineering
- **EDA Insights**:
  - Bike rentals (`cnt`) are positively correlated with temperature (`temp`).
  - High demand is more frequent during working hours and in summer months.
- **Feature Engineering**:
  - Categorical features (`season`, `hour`, etc.) were one-hot encoded.
  - New binary target (`high_demand`) was created for classification.

## Models and Results
### Regression Models
- **Linear Regression**: RMSE = 139.17
- **Random Forest Regressor**: RMSE = 40.29

### Classification Models
- **Default Random Forest Classifier**:
  - Accuracy: 97%
  - Class `1` F1-Score: 79%
- **Tuned Random Forest Classifier**:
  - Accuracy: 98%
  - Class `1` F1-Score: 83%
  - **Best Hyperparameters**: `max_depth=20`, `min_samples_split=10`, `n_estimators=200`


#### Hyperparameter Tuning
- **Method**: GridSearchCV with 3-fold cross-validation.
- **Parameters Tuned**:
  - `n_estimators`: Number of trees in the forest.
  - `max_depth`: Maximum depth of the trees.
  - `min_samples_split`: Minimum samples required to split a node.
- **Best Hyperparameters**:
  - `n_estimators=200`
  - `max_depth=20`
  - `min_samples_split=10`
- **Improvement**:
  - Class `1` F1-Score improved from 79% to 83%.
  - Overall accuracy improved from 97% to 98%.

### Model Comparison

| **Model**              | **Accuracy** | **Precision (Class 1)** | **Recall (Class 1)** | **F1-Score (Class 1)** | **Notes**                           |
|-------------------------|--------------|--------------------------|----------------------|------------------------|-------------------------------------|
| **Decision Tree**       | **98%**      | 83%                      | 84%                  | 84%                    | Slight overfitting. Simpler model.  |
| **Random Forest**       | **98%**      | 84%                      | 81%                  | 83%                    | Balanced performance.               |
| **XGBoost (Default)**   | **99%**      | 92%                      | 88%                  | 90%                    | Best recall for class `1`.          |
| **XGBoost (Tuned)**     | **98%**      | 91%                      | 87%                  | 89%                    | Minor improvements post-tuning.     |

### Decision Tree Visualization

- A single decision tree was trained and visualized to show how the model splits the data.
- **Insights**:
  - Features like `temp`, `hr`, and `workingday` strongly influence predictions.
- Visualization:

(Include the generated tree plot here.)

### XGBoost Tuning

- **Best Parameters**:
  - `learning_rate`: **0.1**
  - `max_depth`: **5**
  - `n_estimators`: **200`
- Tuning slightly improved precision but slightly reduced recall for class `1`.


### Limitations
- While the model achieves strong performance on the test dataset (98% accuracy), it sometimes predicts `0` for test cases that are designed to represent high-demand scenarios. This behavior could be due to:
  - The training data having insufficient high-demand samples, leading to bias toward class `0`.
  - Subtle differences in preprocessing during training and API deployment.
- Future improvements:
  - Addressing class imbalance using oversampling (e.g., SMOTE) or collecting more high-demand examples.
  - Exploring more advanced models like XGBoost or LightGBM to better capture decision boundaries.

### Interpretation of Predictions
- `0`: Indicates low bike demand (<= 500 rentals per hour).
- `1`: Indicates high bike demand (> 500 rentals per hour).
- For certain edge cases, the model may return unexpected results. These are noted in the limitations and serve as areas for future improvement.

-------

## How to Run the Code

[![Watch the demo video](assets/ML-midterm_record.mp4)]
1. Clone the repository.
2. Install dependencies:
   pip install -r requirements.txt
3. run the notebook:
jupyter notebook to see notebook.ipynb

To use the model for predictions, load tuned_random_forest.pkl in the deployment script.
Test it by POSTMAN Post request:
http://127.0.0.1:5000/predict
example input body raw json:
{
    "instant": 1,
    "season": 3,
    "yr": 1,
    "mnth": 7,
    "hr": 18,
    "holiday": 0,
    "weekday": 2,
    "workingday": 1,
    "weathersit": 1,
    "temp": 0.85,
    "atemp": 0.9,
    "hum": 0.3,
    "windspeed": 0.1,
    "day": 15,
    "day_of_week": 4
}

output : 
{
  "prediction": 0
}

---

## **Deployment**
## Deployment
The project includes a Flask API for model deployment. To deploy locally:
1. Start the Flask server:
   python predict.py
Send a POST request with input data to the /predict endpoint.
Example input:
json
{
    "temp": 0.5,
    "atemp": 0.55,
    "hum": 0.6,
    "windspeed": 0.2,
    "season_2": 0,
    "season_3": 1,
    "season_4": 0,
    ...
}
The API will return the predicted bike rentals (regression) or demand classification (classification).

### Docker Testing Issue
While the Dockerfile builds successfully and the container runs, I encountered issues connecting to the `/predict` endpoint locally. The likely causes are related to network binding or JSON payload handling. I plan to address these by further debugging the Flask route and Docker configuration.