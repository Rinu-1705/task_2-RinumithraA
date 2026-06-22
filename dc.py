import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 1. LOAD AND UNDERSTAND THE DATASET
print("="*50)
print("STEP 1: LOADING AND UNDERSTANDING THE DATASET")
print("="*50)

# Load the Iris dataset
iris = load_iris()
X = iris.data  
y = iris.target 

# Convert to DataFrame f
df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = y
df['species_name'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# Explore the dataset
print(f"Dataset shape: {df.shape}")
print(f"\nFirst 5 rows:")
print(df.head())
print(f"\nDataset info:")
print(df.info())
print(f"\nStatistical summary:")
print(df.describe())
print(f"\nClass distribution:")
print(df['species_name'].value_counts())

# Visualize the data
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.scatterplot(data=df, x='sepal length (cm)', y='sepal width (cm)', 
                hue='species_name', style='species_name')
plt.title('Sepal Length vs Width')

plt.subplot(1, 2, 2)
sns.scatterplot(data=df, x='petal length (cm)', y='petal width (cm)', 
                hue='species_name', style='species_name')
plt.title('Petal Length vs Width')

plt.tight_layout()
plt.show()

# 2. SPLIT DATA INTO TRAINING AND TESTING SETS
print("\n" + "="*50)
print("STEP 2: SPLITTING THE DATA")
print("="*50)

# Split the data: 70% training, 30% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Training set size: {len(X_train)} samples")
print(f"Testing set size: {len(X_test)} samples")
print(f"Training set class distribution: {np.bincount(y_train)}")
print(f"Testing set class distribution: {np.bincount(y_test)}")

# 3. APPLY A SIMPLE CLASSIFICATION ALGORITHM
print("\n" + "="*50)
print("STEP 3: TRAINING THE MODEL")
print("="*50)

# Create and train a k-Nearest Neighbors classifier
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

print("Model trained successfully!")
print(f"Model type: k-Nearest Neighbors (k=5)")
print(f"Training set size: {len(X_train)} samples")

# Make predictions on the test set
y_pred = model.predict(X_test)

# 4. EVALUATE THE MODEL
print("\n" + "="*50)
print("STEP 4: EVALUATING THE MODEL")
print("="*50)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
# Convert to percentage for human-friendly display
accuracy_percent = accuracy * 100
print(f"Accuracy: {accuracy_percent:.2f}%")

# Detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Confusion Matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Visualize confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=iris.target_names, 
            yticklabels=iris.target_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# 5. FEATURE STATISTICS
print("\n" + "="*50)
print("STEP 5: FEATURE STATISTICS")
print("="*50)

# Calculate mean and standard deviation for each feature per class
feature_stats = df.groupby('species_name')[iris.feature_names].mean()
print("Mean feature values by species:")
print(feature_stats)

# Visualize feature statistics
plt.figure(figsize=(12, 5))

for idx, feature in enumerate(iris.feature_names, 1):
    plt.subplot(2, 2, idx)
    for species in ['setosa', 'versicolor', 'virginica']:
        species_data = df[df['species_name'] == species][feature]
        plt.hist(species_data, alpha=0.6, label=species, bins=10)
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.title(f'Distribution of {feature}')
    plt.legend()

plt.tight_layout()
plt.show()

# 6. MAKE PREDICTIONS ON NEW DATA
print("\n" + "="*50)
print("STEP 6: MAKING PREDICTIONS")
print("="*50)

# Example: Predict on a new sample
new_sample = np.array([[5.0, 3.5, 1.5, 0.2]])  # Similar to setosa
prediction = model.predict(new_sample)
prediction_proba = model.predict_proba(new_sample)

print(f"New sample: {new_sample[0]}")
print(f"Predicted class: {iris.target_names[prediction[0]]}")
print(f"Prediction probabilities: {prediction_proba[0]}")
print(f"Probability per class: {dict(zip(iris.target_names, prediction_proba[0]))}")
