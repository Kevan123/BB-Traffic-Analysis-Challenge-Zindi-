
# Barbados Traffic Analysis Challenge (Zindi)

A machine learning solution for predicting traffic congestion levels at road camera intersections in Barbados.

**Competition:** Barbados Traffic Analysis Challenge (Zindi)  
**Task:** Multiclass Classification  
**Primary Model:** XGBoost  
**Techniques:** Time-Series Feature Engineering, Hyperparameter Optimization, Model Comparison, and Ensemble Learning  
**Environment:** Python / Google Colab

---

## 1. Project Overview

This repository contains a machine learning solution developed for the **Barbados Traffic Analysis Challenge** hosted on Zindi.

The objective is to predict traffic congestion levels at four camera views covering the Norman Niles intersections in Barbados. Predictions are made for individual one-minute time segments using historical traffic signaling observations.

Two congestion indicators are predicted independently:

- `congestion_enter_rating`: Congestion experienced by vehicles entering an intersection.
- `congestion_exit_rating`: Congestion experienced by vehicles exiting an intersection.

Each prediction belongs to one of four congestion categories.

### Congestion Classification

| Congestion Level | Integer Encoding |
|------------------|------------------|
| Free flowing | 0 |
| Light delay | 1 |
| Moderate delay | 2 |
| Heavy delay | 3 |

### Approach

The solution combines:

1. Historical traffic signaling observations.
2. Rolling-window feature engineering.
3. Temporal separation between observations and prediction targets.
4. XGBoost multiclass classification.
5. Hyperparameter optimization.
6. Comparison against alternative classification algorithms.
7. Probability-based ensemble learning.

The primary approach uses a **15-minute historical observation window** with a **2-minute embargo** before each prediction target.

Separate models are trained to predict entering and exiting congestion levels.

---

## 2. Repository Structure

```text
BB-Traffic-Analysis-Challenge-Zindi/
│
├── Barbados_Traffic_Analysis_Challenge_Zindi.ipynb
│   └── Main analysis and modeling notebook
│
├── README.md
│   └── Project documentation
│
└── Zindi Data/
    ├── Train.csv
    ├── TestInputSegments.csv
    └── SampleSubmission.csv
```

**Note:** The competition datasets are not distributed with this repository. They must be obtained separately from Zindi, subject to the competition's data usage terms.

---

## 3. Requirements

### Python Version

Python 3.8 or later, with compatible versions of the required libraries.

### Dependencies

The project uses the following Python packages:

- pandas
- numpy
- scikit-learn
- xgboost
- jupyter

Install the dependencies using:

```bash
pip install pandas numpy scikit-learn xgboost jupyter
```

### Development Environment

The notebook was developed using **Google Colab**.

It can also be executed locally using Jupyter Notebook, provided that the dataset paths are updated.

**Model fallback:** If XGBoost is unavailable, the notebook includes fallback logic to train a `RandomForestClassifier`. The fallback is not equivalent to XGBoost and may produce different predictions and performance.

---

## 4. Dataset Description

The dataset originates from the Barbados Traffic Analysis Challenge on Zindi.

Three files are required.

| File | Description |
|------|-------------|
| `Train.csv` | Labelled training observations containing 16,076 rows across four camera views and seven dates |
| `TestInputSegments.csv` | Historical traffic observations supplied as context for test prediction windows |
| `SampleSubmission.csv` | Submission template containing prediction identifiers |

### Required Input Columns

The principal columns used for feature construction include:

| Column | Description |
|--------|-------------|
| `view_label` | Identifier for the road camera view |
| `time_segment_id` | Identifier for a one-minute traffic observation segment |
| `signaling` | Recorded traffic signaling level |
| `congestion_enter_rating` | Target classification for entering traffic |
| `congestion_exit_rating` | Target classification for exiting traffic |

The target columns are required for labelled training observations but are not necessarily available for every test-context observation.

Additional columns may be used for temporal features or submission formatting.

### Signaling Categories

The signaling variable is ordinally encoded.

| Signaling Level | Encoding |
|-----------------|----------|
| None | 0 |
| Low | 1 |
| Medium | 2 |
| High | 3 |

This encoding preserves the ordering of signaling intensity.

---

## 5. Configuration

The main configuration parameters are defined near the beginning of the notebook.

```python
TRAIN_FILE = "/content/Train.csv"
TEST_INPUT_FILE = "/content/TestInputSegments.csv"
SAMPLE_SUB_FILE = "/content/SampleSubmission.csv"

SUBMISSION_OUT = "submission.csv"

HIST_LEN = 15
EMBARGO = 2

USE_TEST_INPUT_AS_TRAIN = True
```

### Parameter Description

| Parameter | Description | Default |
|-----------|-------------|---------|
| `TRAIN_FILE` | Location of the training dataset | `/content/Train.csv` |
| `TEST_INPUT_FILE` | Location of the test-context dataset | `/content/TestInputSegments.csv` |
| `SAMPLE_SUB_FILE` | Location of the submission template | `/content/SampleSubmission.csv` |
| `SUBMISSION_OUT` | Output submission filename | `submission.csv` |
| `HIST_LEN` | Number of historical one-minute observations | 15 |
| `EMBARGO` | Gap between the historical window and prediction target | 2 |
| `USE_TEST_INPUT_AS_TRAIN` | Controls whether eligible labelled test-input observations are included in training | `True` |

**Important:** The use of `TestInputSegments.csv` as additional training data must comply with the competition's rules and should only include observations with legitimate, available training labels.

If executing locally, update the dataset paths accordingly.

---

## 6. Methodology

### 6.1 Data Preprocessing

The preprocessing workflow performs the following operations:

1. Loads the training and test-context datasets.
2. Encodes categorical signaling observations.
3. Encodes congestion classifications.
4. Parses temporal information.
5. Sorts observations by camera view and time segment.
6. Creates a unified observation dataset for historical feature lookup.

The training and test-context observations are combined into an intermediate dataset named `all_df`.

This allows historical observations from both sources to support feature construction.

The test-context data is used as observed historical information, not as a source of unknown future congestion labels.

### 6.2 Rolling-Window Feature Engineering

For a target observation at time segment `T` and camera `C`, historical signaling information is extracted from the same camera.

The historical window is defined as:

```text
[T - EMBARGO - HIST_LEN, ..., T - EMBARGO - 1]
```

With the default configuration:

```text
HIST_LEN = 15
EMBARGO  = 2
```

The observation window covers:

```text
T-17, T-16, ..., T-4, T-3
```

The target segment is:

```text
T
```

Segments `T-2` and `T-1` are excluded from the feature window.

This creates a two-minute embargo immediately before the prediction target.

**Purpose:** The embargo ensures that features are constructed from sufficiently earlier observations, reducing the risk of using information too close to the prediction target.

### 6.3 Feature Categories

The feature engineering process constructs several groups of predictors.

#### A. Lag Features

Individual signaling observations from the historical window:

```text
sig_lag_1
sig_lag_2
sig_lag_3
...
sig_lag_15
```

These preserve information about changes in signaling intensity over time.

#### B. Statistical Aggregates

Summary statistics computed over the historical window:

- Mean signaling intensity
- Standard deviation
- Minimum signaling intensity
- Maximum signaling intensity

#### C. Advanced Statistical Features

Additional features introduced during later experiments:

- Median
- Variance
- Linear trend slope
- Trend difference

These capture variation and directional changes in recent signaling activity.

#### D. Temporal Features

Time-related variables include:

- Hour of day
- Day of week
- Month
- Cyclical time-of-day encoding

Cyclical encoding is implemented using sine and cosine transformations:

```python
tod_sin = np.sin(2 * np.pi * hour / 24)
tod_cos = np.cos(2 * np.pi * hour / 24)
```

This represents the cyclical relationship between hours of the day.

#### E. Interaction Features

An interaction term combines historical signaling intensity with time of day:

```python
sig_mean_15 * tod_sin
```

This feature allows the model to capture potential relationships between signaling intensity and daily traffic patterns.

#### F. Camera Identification

Camera-specific information is represented through one-hot encoding of `view_label`.

This allows the model to account for differences between camera locations.

### 6.4 Handling Incomplete Historical Windows

At the beginning of an observation series, fewer than 15 historical segments may be available.

The notebook addresses this by left-padding incomplete windows using the earliest available historical value.

This avoids introducing future observations into the historical feature window.

However, the approach assumes that at least one eligible historical value exists and that the available observations are appropriately ordered.

---

## 7. Model Development

### 7.1 Primary Model: XGBoost

XGBoost is used as the primary classification algorithm.

Two independent classifiers are trained:

1. Entering traffic congestion classifier.
2. Exiting traffic congestion classifier.

The baseline configuration is:

```python
XGBClassifier(
    objective="multi:softprob",
    num_class=4,
    eval_metric="mlogloss",
    max_depth=4,
    n_estimators=400,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    n_jobs=-1
)
```

The `multi:softprob` objective generates predicted class probabilities for the four congestion categories.

### 7.2 Hyperparameter Optimization

Two hyperparameter optimization strategies are explored.

#### GridSearchCV

The initial search explores combinations of:

- `max_depth`
- `n_estimators`
- `learning_rate`
- `subsample`
- `colsample_bytree`

Grid search evaluates candidate combinations using cross-validation.

#### RandomizedSearchCV

An expanded search space additionally considers:

- `gamma`
- `min_child_weight`

The randomized search configuration uses:

```python
n_iter = 100
cv = 3
```

Randomized search samples candidate configurations rather than exhaustively evaluating every combination.

**Recommendation:** RandomizedSearchCV is generally more computationally practical than exhaustive grid search for the expanded parameter space.

**Validation consideration:** Because observations are temporal and camera-dependent, ordinary shuffled or random cross-validation may produce optimistic performance estimates. Chronological or grouped temporal validation is preferable for assessing generalization.

---

## 8. Comparison Models

The notebook includes three alternative classification approaches:

- Logistic Regression
- Support Vector Classifier (SVC)
- Random Forest Classifier

For the SVC experiments, probability estimation is enabled using:

```python
SVC(probability=True)
```

The models are compared using weighted F1-score and accuracy.

### 8.1 In-Sample Training Results

The following results are calculated using training observations.

| Model | Target | Weighted F1 | Accuracy |
|-------|--------|------------:|---------:|
| XGBoost | Enter | 0.701 | 0.739 |
| XGBoost | Exit | 0.950 | 0.963 |
| Logistic Regression | Enter | 0.584 | 0.646 |
| Logistic Regression | Exit | 0.935 | 0.956 |
| SVC | Enter | 0.645 | 0.705 |
| SVC | Exit | 0.935 | 0.956 |
| Random Forest | Enter | 0.998 | 0.998 |
| Random Forest | Exit | 1.000 | 1.000 |

**Important:** These are in-sample results and should not be interpreted as estimates of performance on unseen observations.

The near-perfect Random Forest scores indicate that the model fits the training observations extremely closely. They do not, by themselves, demonstrate generalization or establish overfitting.

A comparison with independent validation or leaderboard results is required to assess generalization.

### 8.2 Zindi Leaderboard Results

The following scores were recorded for the submitted model configurations.

| Model | Overall F1 | Accuracy |
|-------|-----------:|---------:|
| **Optimized XGBoost** | **0.7408** | **0.7818** |
| Random Forest | 0.7342 | 0.7795 |
| XGBoost (Baseline) | 0.7218 | 0.7773 |

### Observations

- Optimized XGBoost achieved the highest reported leaderboard F1-score.
- Random Forest performed well but did not exceed optimized XGBoost.
- Hyperparameter optimization improved the leaderboard F1-score relative to baseline XGBoost.
- The difference between in-sample and leaderboard performance illustrates the importance of evaluating models on unseen data.

Based on these results, **optimized XGBoost was the strongest reported individual model**.

---

## 9. Ensemble Learning

The notebook also explores probability-based ensemble learning.

Predicted class probabilities from the tuned XGBoost classifier and Random Forest classifier are averaged.

```python
avg_proba = (proba_xgb[i] + proba_rf[i]) / 2
```

The predicted class is selected from the averaged probability distribution.

```python
predicted_class = np.argmax(avg_proba)
```

### Ensemble Rationale

The ensemble aims to combine potentially complementary prediction patterns from two different tree-based learning algorithms.

XGBoost uses sequential gradient boosting, whereas Random Forest uses bagging across independently constructed decision trees.

Averaging predicted probabilities may reduce the influence of model-specific errors.

However, ensemble performance depends on the quality, calibration, and diversity of the individual models.

An ensemble is not guaranteed to outperform its strongest component.

### Ensemble Output

The ensemble submission is written to:

```text
submission_ensemble_random.csv
```

The reported leaderboard comparison above does not establish whether this ensemble outperformed the optimized XGBoost model.

---

## 10. Submission Generation

The notebook includes a helper function:

```python
generate_submission_file()
```

This function handles prediction generation and submission formatting.

### Submission Structure

The generated submission contains the following columns:

| Column | Description |
|--------|-------------|
| `ID` | Prediction identifier |
| `Target` | Predicted congestion category |
| `Target_Accuracy` | Predicted congestion category, following the submission format implemented in the notebook |

An example of the submission structure is:

```csv
ID,Target,Target_Accuracy
example_001,free flowing,free flowing
example_002,light delay,light delay
example_003,moderate delay,moderate delay
example_004,heavy delay,heavy delay
```

These are illustrative identifiers rather than actual competition submission records.

**Note:** In the final submission implementation, `Target_Accuracy` is populated with the predicted category label rather than a confidence probability.

Earlier notebook experiments considered confidence-based values for this column.

Always validate the generated file against the official `SampleSubmission.csv` format before submitting.

---

## 11. Running the Notebook

### Option A: Google Colab

Google Colab is the recommended execution environment.

**Step 1: Open the notebook**

Upload:

```text
Barbados_Traffic_Analysis_Challenge_Zindi.ipynb
```

to Google Colab.

**Step 2: Upload competition data**

Upload these files to the Colab working directory:

```text
/content/Train.csv
/content/TestInputSegments.csv
/content/SampleSubmission.csv
```

Alternatively, mount Google Drive and update the configuration paths.

**Step 3: Confirm configuration**

Ensure the file paths in the notebook match the uploaded files.

**Step 4: Execute notebook cells**

Select:

```text
Runtime > Run all
```

**Step 5: Generate predictions**

Run the submission generation cells.

**Step 6: Download submission**

Download the generated CSV from the Colab file browser.

### Option B: Local Jupyter Notebook

Install the required packages:

```bash
pip install pandas numpy scikit-learn xgboost jupyter
```

Launch the notebook:

```bash
jupyter notebook Barbados_Traffic_Analysis_Challenge_Zindi.ipynb
```

Update the file paths in the configuration block.

For example:

```python
TRAIN_FILE = "./data/Train.csv"
TEST_INPUT_FILE = "./data/TestInputSegments.csv"
SAMPLE_SUB_FILE = "./data/SampleSubmission.csv"
```

Execute the notebook cells in sequence.

### Computational Considerations

Hyperparameter optimization can be computationally expensive, particularly when using multiple cross-validation folds.

For a faster baseline reproduction:

1. Run preprocessing and feature engineering.
2. Train the baseline XGBoost models.
3. Skip the GridSearchCV and RandomizedSearchCV experiments.
4. Generate the baseline submission.

This avoids the computational overhead of repeated model fitting.

---

## 12. Known Issues and Limitations

### 12.1 Hardcoded Colab File Paths

The default configuration uses `/content/` file paths.

These paths must be modified when running outside Google Colab.

### 12.2 In-Sample Model Evaluation

The primary model comparison table reports training-set performance.

Training metrics can overestimate performance on unseen data.

For future improvements, use:

- Chronological train-validation splits.
- Time-aware cross-validation.
- Separate evaluation periods.
- Camera-level robustness checks where appropriate.

### 12.3 Notebook State Dependencies

The notebook stores intermediate datasets, feature matrices, models, and evaluation results in memory.

Examples include:

```python
evaluation_results
X_enter
X_exit
X_sub
```

Running cells out of sequence can result in missing or inconsistent objects.

Some later cells contain recovery logic that reinitializes models or reconstructs metrics.

For consistent reproduction, executing the notebook sequentially in a fresh runtime is recommended.

### 12.4 Multiple Feature Engineering Configurations

Later notebook cells redefine the feature generation function:

```python
make_feature_vector()
```

These cells introduce additional statistical and trend features.

They also reconstruct the feature matrices:

```python
X_enter
X_exit
X_sub
```

Models trained before and after this reconstruction may use different feature sets.

**Important:** Models and feature matrices from different feature engineering configurations should not be mixed.

### 12.5 Test-Context Data Usage

The notebook combines training and test-context observations to construct historical features.

This is appropriate only when the historical observations are legitimately available at prediction time.

Feature construction should avoid:

- Using future observations.
- Accessing hidden test labels.
- Including observations that would not be available during deployment.
- Introducing information leakage through preprocessing or model validation.

### 12.6 Submission Column Interpretation

The `Target_Accuracy` column contains class labels in the final notebook implementation.

It should not be interpreted as a prediction confidence score.

The output schema should be checked against the official competition submission template.

### 12.7 Model Generalization

The dataset covers a limited number of camera views and observation dates.

Performance may differ across:

- Unseen dates.
- Different traffic patterns.
- Other intersections.
- Different camera installations.
- Changes in road conditions or traffic management.

Additional temporal and spatial validation would be required before considering the model suitable for operational deployment.

---

## 13. Potential Improvements

Several extensions could improve the analytical robustness and practical utility of the solution.

### Model Validation

- Implement time-series cross-validation.
- Evaluate performance by camera view.
- Compare weekday and weekend performance.
- Analyze performance across peak and off-peak periods.
- Report class-level precision, recall, and F1-score.
- Examine confusion matrices for different congestion levels.

### Feature Engineering

- Introduce additional historical window lengths.
- Explore exponentially weighted historical aggregates.
- Develop features measuring short-term signaling volatility.
- Evaluate additional lag interactions.
- Investigate changes in traffic patterns across consecutive periods.

### Model Development

- Compare XGBoost with LightGBM and CatBoost.
- Evaluate class imbalance handling.
- Explore probability calibration.
- Optimize ensemble weights using validation data.
- Assess ordinal classification approaches that account for the ordered congestion categories.

### Explainability

- Apply SHAP to identify influential features.
- Examine feature importance across camera views.
- Evaluate whether historical signaling features behave consistently across prediction periods.

### Reproducibility

- Refactor feature engineering into reusable Python functions.
- Separate training, validation, and inference workflows.
- Save fitted models and feature schemas.
- Record exact package versions.
- Introduce automated checks for feature and submission consistency.

---

## 14. Key Findings

The main findings from the reported experiments are:

1. Historical traffic signaling observations provide useful predictive information for congestion classification.

2. Rolling-window features allow the models to capture recent signaling conditions and temporal patterns.

3. Optimized XGBoost achieved the strongest reported individual-model leaderboard performance, with an overall F1-score of **0.7408** and accuracy of **0.7818**.

4. Random Forest achieved near-perfect training scores but did not outperform optimized XGBoost on the reported leaderboard metrics.

5. Hyperparameter optimization improved the reported leaderboard F1-score compared with baseline XGBoost.

6. Probability averaging provides an additional ensemble approach, although its comparative leaderboard performance is not established by the results presented here.

7. Time-aware validation and more systematic experiment tracking would strengthen the reliability and reproducibility of future work.

---

## 15. Data Usage and Disclaimer

This project is shared for educational, research, and reference purposes.

The underlying competition datasets belong to the competition organizers and their respective data providers.

Dataset access, redistribution, and usage are subject to the applicable Zindi competition terms.

Model performance figures reflect the experiments reported in the project notebook and should not be interpreted as guaranteed performance in operational traffic management systems.

---

## 16. Project Summary

| Component | Implementation |
|-----------|----------------|
| Competition | Barbados Traffic Analysis Challenge |
| Platform | Zindi |
| Problem Type | Multiclass Classification |
| Number of Classes | 4 |
| Prediction Targets | Entering and exiting congestion |
| Primary Algorithm | XGBoost |
| Comparison Algorithms | Logistic Regression, SVC, Random Forest |
| Historical Window | 15 minutes |
| Embargo Period | 2 minutes |
| Feature Types | Lag, statistical, temporal, trend, camera identification |
| Optimization | GridSearchCV, RandomizedSearchCV |
| Ensemble | Probability averaging |
| Best Reported Model | Optimized XGBoost |
| Best Reported Overall F1 | 0.7408 |
| Best Reported Accuracy | 0.7818 |
| Development Environment | Google Colab |
| Programming Language | Python |

---

**Project:** Barbados Traffic Analysis Challenge  
**Platform:** Zindi  
**Implementation:** Python, Scikit-learn, XGBoost
