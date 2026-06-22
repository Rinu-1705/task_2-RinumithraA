# Iris Dataset Classification with k-Nearest Neighbors

**dc.py** is a comprehensive machine learning project that demonstrates the complete workflow of building, training, and evaluating a classification model. It uses the famous **Iris dataset** and implements a **k-Nearest Neighbors (KNN)** classifier to predict iris flower species based on their physical measurements.

This script is ideal for learning machine learning fundamentals including data exploration, model training, evaluation, and prediction.

---

## Project Purpose

The script performs end-to-end machine learning classification with 6 main stages:

1. **Data Loading & Exploration** - Understand the dataset
2. **Data Splitting** - Prepare training and testing sets
3. **Model Training** - Train a KNN classifier
4. **Model Evaluation** - Assess performance metrics
5. **Feature Analysis** - Identify important features
6. **Prediction** - Make predictions on new data

---

## Dataset: Iris Flowers

### Dataset Characteristics

- **Total Samples**: 150 iris flower records
- **Classes**: 3 species (Setosa, Versicolor, Virginica)
- **Features**: 4 physical measurements per flower
- **Feature Names**:
  - `sepal length (cm)` - Length of the flower's outer petals
  - `sepal width (cm)` - Width of the flower's outer petals
  - `petal length (cm)` - Length of the flower's inner petals
  - `petal width (cm)` - Width of the flower's inner petals

### Target Classes

| Class ID | Species Name | Description |
|----------|--------------|-------------|
| 0 | Setosa | Small iris with short measurements |
| 1 | Versicolor | Medium-sized iris |
| 2 | Virginica | Larger iris with longer measurements |

Each species has 50 samples, making it a balanced dataset.

---

## Code Structure & Workflow

### Step 1: Load and Understand the Dataset

```python
# Load Iris dataset from sklearn
iris = load_iris()
X = iris.data  # Features (150 x 4)
y = iris.target  # Target labels (150,)

# Create DataFrame for better visualization
df = pd.DataFrame(X, columns=iris.feature_names)
df['species_name'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
```

**Outputs**:
- Dataset shape and first 5 rows
- Statistical summary (mean, std, min, max)
- Class distribution (count per species)
- 2 scatter plots:
  - Sepal Length vs Width
  - Petal Length vs Width

### Step 2: Split Data into Training and Testing Sets

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
```

**Data Split Breakdown**:
- **Training Set**: 70% (105 samples) - Used to train the model
- **Testing Set**: 30% (45 samples) - Used to evaluate the model
- **Stratification**: Ensures balanced class distribution in both sets

**Outputs**:
- Training/testing set sizes
- Class distribution in each set

### Step 3: Train the Classification Model

```python
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
```

**Algorithm**: k-Nearest Neighbors (KNN)
- **How it works**: Classifies based on the 5 nearest neighbors in the training set
- **Parameter**: k=5 (checks 5 closest neighbors)
- **Training**: Simply stores the training data (lazy learner)

**Outputs**:
- Model successfully trained
- Model configuration details

### Step 4: Evaluate the Model

```python
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
```

**Evaluation Metrics**:

1. **Accuracy**: Percentage of correct predictions
   - Formula: (Correct Predictions / Total Predictions) × 100
   
2. **Classification Report**: Detailed per-class metrics
   - **Precision**: Accuracy of positive predictions
   - **Recall**: Coverage of actual positive class
   - **F1-Score**: Harmonic mean of precision and recall
   - **Support**: Number of samples per class

3. **Confusion Matrix**: Grid showing prediction results
   - Rows: Actual classes
   - Columns: Predicted classes
   - Diagonal: Correct predictions

**Visualizations**:
- Heatmap of confusion matrix showing correct/incorrect predictions

### Step 5: Feature Importance Analysis

```python
feature_importance = pd.DataFrame({
    'feature': iris.feature_names,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
```

**Purpose**: Identifies which features are most useful for classification

**Output**: Bar chart ranking features by importance

### Step 6: Make Predictions on New Data

```python
new_sample = np.array([[5.0, 3.5, 1.5, 0.2]])
prediction = model.predict(new_sample)
prediction_proba = model.predict_proba(new_sample)
```

**Demonstrates**:
- Making predictions on custom samples
- Predicted class
- Confidence probabilities for each class

**Example**:
- Input: `[5.0, 3.5, 1.5, 0.2]` (sepal length, sepal width, petal length, petal width)
- Output: Predicted species (e.g., "setosa")

---

## Running the Script

### Basic Execution

```bash
python dc.py
```

### Expected Output

1. **Console Output**:
   - Dataset information and statistics
   - Data split details
   - Model training confirmation
   - Accuracy percentage
   - Classification report with precision/recall/F1-scores
   - Confusion matrix
   - Feature importance ranking
   - Predictions on sample data

2. **Visualizations** (Opens in separate windows):
   - Sepal and petal scatter plots
   - Confusion matrix heatmap
   - Feature importance bar chart

---

## Key Performance Metrics Explained

### Accuracy
- **Definition**: Percentage of correct predictions
- **Formula**: (TP + TN) / (TP + TN + FP + FN)
- **Typical Result**: 90-98% on Iris dataset
- **Interpretation**: Higher is better

### Precision
- **Definition**: Of predicted positive cases, how many were correct
- **Formula**: TP / (TP + FP)
- **Use Case**: When false positives are costly

### Recall (Sensitivity)
- **Definition**: Of actual positive cases, how many were found
- **Formula**: TP / (TP + FN)
- **Use Case**: When false negatives are costly

### F1-Score
- **Definition**: Balanced average of precision and recall
- **Formula**: 2 × (Precision × Recall) / (Precision + Recall)
- **Use Case**: When you need balance between precision and recall

---

## Understanding the Confusion Matrix

```
                Predicted
              Setosa  Versicolor  Virginica
Actual Setosa    15       0           0
       Versicolor 0      14           1
       Virginica  0       1          14
```

**Reading the Matrix**:
- **Diagonal values** (15, 14, 14): Correct predictions
- **Off-diagonal values**: Misclassifications
- **Row sum**: Total samples in each actual class
- **Column sum**: Total predictions for each class

---

## How k-Nearest Neighbors (KNN) Works

### Algorithm Steps

1. **Training Phase**: Store all training samples
2. **Prediction Phase**:
   - For a new sample, calculate distance to all training samples
   - Find the 5 nearest neighbors (k=5)
   - Use majority vote of these 5 neighbors
   - Assign the most common class

### Distance Metric
- Default: Euclidean distance
- Formula: √((x₁-x₂)² + (y₁-y₂)² + ...)

### Why k=5?
- Balances between overfitting and underfitting
- Odd number prevents ties in voting
- Common choice for most datasets
- Can be tuned based on dataset size

---

