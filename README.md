# car-pred

# # Car Rental Price Prediction Model Evaluation Report

# ## Executive Summary

# This report presents the evaluation results of an XGBoost regression model developed to predict daily rental costs for vehicles. The model achieves exceptional performance with an R² score of 0.98, indicating that it successfully captures 98% of the variance in rental prices. With a Root Mean Square Error (RMSE) of 27.73 SAR and Mean Absolute Error (MAE) of 8.07 SAR, the model demonstrates strong predictive capability for this business use case.

# **Key Performance Metrics:**
# - **RMSE:** 27.73 SAR
# - **MAE:** 8.07 SAR
# - **R²:** 0.98
# - **Number of Features:** 55
# - **Training Sample Size:** 11,917,282

# ## Data Overview

# The model was trained on a large-scale dataset containing car rental contract information with the following characteristics:

# - **Data Source:** reema_table.csv
# - **Total Records:** 11,917,282
# - **Target Variable:** total_daily_rent_cost
# - **Key Features Categories:**
#   - Contract details (type, status, duration)
#   - Vehicle information (maker, model, manufacture year)
#   - Temporal features (contract start/end dates)
#   - Registration and insurance information

# ## Preprocessing Methodology

# ### Data Loading and Memory Optimization

# - **Custom Data Types:** Implemented category and smaller numeric dtypes to reduce memory footprint
# - **Essential Column Selection:** Loaded only the columns required for analysis
# - **Batch Processing:** Data processed in chunks of 200,000 rows to manage memory constraints

# ### Data Cleaning

# - **Missing Values:** Removed rows with missing target values
# - **Outlier Handling:** Applied IQR-based outlier detection and capping on the target variable
#   - Lower bound: 1st percentile - 1.5 × IQR
#   - Upper bound: 99th percentile + 1.5 × IQR
# - **NaN and Infinity Value Treatment:** Replaced NaN values with zero and infinite values with large finite numbers

# ### Feature Engineering

# 1. **Date Processing:**
#    - Extracted year, month, day, and day of week from contract dates
#    - Created cyclical encoding for month and day of week using sine and cosine transformations
#    - Added weekend flags for start and end dates

# 2. **Contract Features:**
#    - Calculated contract duration in days
#    - Identified duration discrepancy between stated and calculated periods

# 3. **Vehicle Characteristics:**
#    - Derived car maker and model popularity based on frequency counts
#    - Calculated vehicle age from manufacture year

# 4. **Categorical Encoding:**
#    - Applied one-hot encoding to contract_type, contract_status, registration_type, and insurance_type
#    - Created binary flags for top 15 car makers

# 5. **Price Features:**
#    - Calculated total cost based on daily rate and rental period

# ## Model Architecture and Training

# ### XGBoost Configuration

# ```python
# XGBRegressor(
#     n_estimators=100,           # Number of boosting rounds
#     learning_rate=0.05,         # Step size shrinkage to prevent overfitting
#     max_depth=7,                # Maximum tree depth
#     subsample=0.8,              # Fraction of samples used for training trees
#     colsample_bytree=0.8,       # Fraction of features used for training trees
#     random_state=42,            # Random seed for reproducibility
#     gpu_id=0,                   # GPU device ID
#     tree_method='gpu_hist',     # Tree construction algorithm using GPU
#     predictor='gpu_predictor',  # Prediction algorithm using GPU
#     missing=np.nan              # Missing value representation
# )
# ```

# ### Training Methodology

# - **GPU Acceleration:** Utilized GPU for model training when available
# - **Memory Management:** Implemented adaptive memory usage based on available GPU resources
# - **Batch Training:** For datasets with >5 million rows, employed incremental batch training
# - **Early Stopping:** Implemented validation-based early stopping to prevent overfitting
# - **Train-Test Split:** 80% training, 20% testing with stratified sampling

# ## Evaluation Results

# ### Performance Metrics

# | Metric | Value | Interpretation |
# |--------|-------|----------------|
# | RMSE | 27.73 SAR | Average prediction error in original units |
# | MAE | 8.07 SAR | Average absolute prediction error |
# | R² | 0.98 | 98% of variance explained by the model |

# ### Performance Analysis

# The model demonstrates exceptional predictive power with an R² of 0.98, which is considered excellent for regression tasks. The difference between RMSE (27.73) and MAE (8.07) indicates the presence of some larger errors in the predictions, likely for high-value rentals or unusual cases.

# For context, with an RMSE of 27.73 SAR:
# - For a typical rental costing 200 SAR/day, the model's prediction would be within ~14% of the actual price
# - The MAE of 8.07 SAR suggests that for most predictions, the error is much smaller (around 4% for a 200 SAR/day rental)

# ### Training Timeline

# - **Training Date:** February 27, 2025
# - **Validation Progress:** 
#   - Validation RMSE at iteration 98: 27.96
#   - Validation RMSE at iteration 99: 27.84
#   - Final Test RMSE: 27.73

# ## Inferred Feature Importance

# While the exact feature importance values are not provided in the output, based on the nature of the model and domain knowledge, we can infer the following likely important features:

# 1. **Vehicle Characteristics:**
#    - Car maker and model (premium brands likely command higher prices)
#    - Vehicle age (newer vehicles typically cost more to rent)
#    - Car maker popularity (indicates market demand)

# 2. **Contract Features:**
#    - Contract duration (longer contracts may have discounted daily rates)
#    - Contract type (different pricing structures for different contract types)

# 3. **Temporal Features:**
#    - Month of contract (seasonal variations in pricing)
#    - Weekend vs. weekday rentals (higher demand during weekends)

# 4. **Insurance and Registration:**
#    - Insurance type (comprehensive insurance costs more)
#    - Registration type (affects operational costs)

# ## Conclusion and Recommendations

# ### Key Strengths

# 1. **Exceptional Accuracy:** The R² of 0.98 indicates an extremely strong model fit.
# 2. **Scalable Processing:** The batch processing approach allows handling of very large datasets.
# 3. **Hardware Optimization:** Adaptive use of GPU resources maximizes performance.

# ### Areas for Improvement

# 1. **Error Distribution Analysis:** Investigate the cases with the largest prediction errors to identify patterns.
# 2. **Feature Expansion:** Consider adding external factors like seasonality, holidays, and local events.
# 3. **Model Interpretability:** Implement SHAP values or feature importance analysis to better understand model decisions.
# 4. **Hyperparameter Tuning:** Consider formal hyperparameter optimization via grid or random search.

# ### Business Impact

# With an MAE of approximately 8 SAR, this model provides highly accurate daily rate predictions that can be used to:
# - Optimize pricing strategies
# - Identify underpriced or overpriced vehicles
# - Forecast revenue with high confidence
# - Support dynamic pricing implementations

# ### Next Steps

# 1. **Model Deployment:** Package the model for production use with appropriate API endpoints.
# 2. **Monitoring System:** Implement drift detection to identify when model retraining is needed.
# 3. **A/B Testing:** Compare pricing recommendations against business-as-usual approaches.
# 4. **Expanded Features:** Investigate additional data sources that might further improve accuracy.

# ---

# *Report generated on: February 27, 2025*

# *Model version: xgboost_model_v1.0.0*
