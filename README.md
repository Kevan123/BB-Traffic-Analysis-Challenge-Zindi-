# BB-Traffic-Analysis-Challenge-Zindi-


Competition: Barbados Traffic Analysis Challenge (Zindi)  
Task: Multiclass classification — predict congestion levels at road camera intersections  

---
Overview
This solution predicts traffic congestion ratings (`congestion_enter_rating` and `congestion_exit_rating`) for four camera views at Norman Niles intersections in Barbados. Each prediction covers a one-minute time segment and is classified into one of four congestion states:
Label	Integer Encoding
free flowing	0
light delay	1
moderate delay	2
heavy delay	3
The core approach builds a rolling-window feature set from historical signaling data per camera, with a 2-minute embargo before each prediction target, and trains separate XGBoost classifiers for enter and exit congestion.
---
Repository Structure
```
.
├── Barbados_Traffic_Analysis_Challenge_Zindi.ipynb   # Main notebook
├── Train.csv                                          # Training data (from Zindi)
├── TestInputSegments.csv                              # Test input data (from Zindi)
├── SampleSubmission.csv                               # Submission template (from Zindi)

```
---
Requirements
Python version
Python 3.8 or later.
Dependencies
```bash
pip install pandas numpy scikit-learn xgboost
```
> **Note:** XGBoost is required for the primary model. If it is not available, the notebook falls back automatically to a `RandomForestClassifier` with comparable depth settings.
The notebook was developed and run on Google Colab. All cells are compatible with a standard Colab environment without additional configuration.
---
Data
The dataset is sourced from the Zindi competition and is not included in this repository. Download the following three files from the competition page and place them in the same directory as the notebook (or update the file paths in the config block):
File	Description
`Train.csv`	Labelled training segments (16,076 rows, 4 camera views, 7 dates)
`TestInputSegments.csv`	Historical context provided for test prediction windows
`SampleSubmission.csv`	Template defining the IDs to predict
Required columns
Both `Train.csv` and `TestInputSegments.csv` must contain at minimum: `view_label`, `time_segment_id`, `signaling`.
---
Configuration
At the top of the notebook, a config block controls the key parameters:
```python
TRAIN_FILE        = "/content/Train.csv"
TEST_INPUT_FILE   = "/content/TestInputSegments.csv"
SAMPLE_SUB_FILE   = "/content/SampleSubmission.csv"
SUBMISSION_OUT    = "submission.csv"

HIST_LEN              = 15    # History window length (number of 1-minute segments)
EMBARGO               = 2     # Embargo gap (minutes) between history window and prediction target
USE_TEST_INPUT_AS_TRAIN = True  # Whether to use TestInputSegments labels during training
```
If running locally (not on Colab), update the file path constants to point to your local data directory.
---
How It Works
1. Data preprocessing
Categorical signaling levels (`none`, `low`, `medium`, `high`) and congestion labels are ordinally encoded to integers. Timestamps are parsed for time-of-day features.
The training set and test input set are concatenated into a unified `all_df`, sorted per camera by `time_segment_id`. This allows the feature builder to look up history across both sets — reflecting the fact that test input segments provide real observed context at inference time.
2. Feature construction
For each prediction target at segment ID `T` and camera `C`, the feature vector is built from the `HIST_LEN` segments immediately preceding `T - EMBARGO` from the same camera:
```
history window: [T - EMBARGO - HIST_LEN, ..., T - EMBARGO - 1]
```
Features extracted from this window:
Lag features — `sig_lag_1` through `sig_lag_15` (most recent to oldest signal encoding)
Aggregates — mean, std, min, max of signal over the window
Advanced aggregates (later cells) — median, variance, linear trend slope, and trend difference
Time-of-day — sine and cosine of the hour (cyclic encoding), day of week, month
Interaction term — `sig_mean_15 * tod_sin`
Camera one-hot — binary column per `view_label`
If the history window is shorter than `HIST_LEN` (beginning of a series), it is left-padded with the earliest available value — a real-time safe strategy that avoids look-ahead.
3. Model training
Two separate classifiers are trained: one for `congestion_enter_rating` and one for `congestion_exit_rating`. The primary model is XGBoost:
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
    n_jobs=-1,
)
```
4. Hyperparameter tuning
The notebook explores two tuning strategies:
GridSearchCV — initial 3×3×3 grid over `max_depth`, `n_estimators`, `learning_rate`, `subsample`, `colsample_bytree`
RandomizedSearchCV — expanded grid (adds `gamma`, `min_child_weight`), 100 iterations sampled, 3-fold CV
> GridSearch over the expanded space is computationally expensive. On Colab, the RandomizedSearch approach is recommended. Allow approximately 30–90 minutes depending on dataset size and available hardware.
>
> 
5. Comparison models
For reference, the notebook also trains and evaluates:
Logistic Regression
Support Vector Classifier (SVC with `probability=True`)
Random Forest Classifier
Training metrics (in-sample weighted F1 / accuracy):
Model	Target	F1	Accuracy
XGBoost	Enter	0.701	0.739

XGBoost	Exit	0.950	0.963
Logistic Regression	Enter	0.584	0.646
Logistic Regression	Exit	0.935	0.956
SVC	Enter	0.645	0.705
SVC	Exit	0.935	0.956
Random Forest	Enter	0.998	0.998
Random Forest	Exit	1.000	1.000
> **Note:** Random Forest shows near-perfect in-sample scores due to overfitting on training data. The XGBoost generalisation to the held-out test set was superior, which is why it was selected as the winning submission. Always interpret in-sample metrics with caution — use the platform leaderboard or a proper held-out split for actual model selection.
Platform (leaderboard) scores:
Model	F1 (Overall)	Accuracy
Optimized XGBoost	0.7408	0.7818
Random Forest	0.7342	0.7795
XGBoost (baseline)	0.7218	0.7773
>
> 
6. Ensemble
The final cells blend predicted probabilities from the best XGBoost (RandomizedSearch-tuned) and the Random Forest by simple averaging:
```python
avg_proba = (proba_xgb[i] + proba_rf[i]) / 2
```
The ensemble submission file is `submission_ensemble_random.csv`.
7. Submission generation
The `generate_submission_file()` helper function wraps the prediction and output formatting logic. It produces a CSV with columns `ID`, `Target`, and `Target_Accuracy`, where both target columns contain the predicted congestion label string.
---
Running the Notebook
On Google Colab (recommended)
Upload the notebook to Colab.
Upload `Train.csv`, `TestInputSegments.csv`, and `SampleSubmission.csv` to `/content/` (the default Colab working directory), or mount Google Drive and update the path constants.
Run all cells in order (Runtime → Run all).
Download the output submission CSV(s) from the file browser panel.
> Cells for GridSearchCV and RandomizedSearchCV (the hyperparameter tuning sections) will take significant time. Skip them and use the base XGBoost models if you only need a quick reproduction of the baseline submission.
Locally
```bash
pip install pandas numpy scikit-learn xgboost jupyter
jupyter notebook Barbados_Traffic_Analysis_Challenge_Zindi.ipynb
```
Update `TRAIN_FILE`, `TEST_INPUT_FILE`, and `SAMPLE_SUB_FILE` in the config block to your local paths before running.
---
Known Issues and Notes
Colab path hardcoding: File paths are hardcoded to `/content/`. If running outside Colab, update the config block.
In-sample evaluation only: The training metrics computed in the notebook are in-sample (models evaluated on data they were trained on). This overestimates generalisation performance, especially for Random Forest. Use cross-validation scores or the platform leaderboard for reliable estimates.
`Target_Accuracy` column: The submission format includes a `Target_Accuracy` column which, per competition specification, is set to the predicted class label string (same as `Target`). This is intentional — earlier iterations computed class confidence probability for this column, but the label-based approach was adopted for the final submissions.
State dependency across cells: The notebook accumulates state (e.g., `evaluation_results`, model variables) across cells. If cells are run out of order or the kernel is restarted mid-run, some later cells include recovery logic to re-initialise models and metrics. These cells are self-contained and will re-train from scratch if needed.
RandomizedSearch + expanded feature rebuild: The advanced feature engineering cells redefine `make_feature_vector` and rebuild `X_enter`, `X_exit`, and `X_sub`. Models trained before these cells use the original feature set; models after use the expanded set. Do not mix models across these two feature regimes.
---
Licence
This solution is shared for educational and reference purposes. The underlying data belongs to the competition organisers. Refer to the Zindi competition terms for data usage restrictions.
