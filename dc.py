import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# STEP 1: Load the Iris dataset
print("\n1. LOADING DATA")
iris = load_iris()
X = iris.data  # Features (4 measurements)
y = iris.target  # Target (species type)

df = pd.DataFrame(X, columns=iris.feature_names)
df['species_name'] = [iris.target_names[i] for i in y]

print(f"Dataset: {df.shape[0]} samples, {df.shape[1]-1} features")
print(f"Species: {list(iris.target_names)}")
print(f"\nFirst 3 rows:\n{df.head(3)}")

# STEP 2: Split data into training and testing sets (70-30 split)
print("\n2. SPLITTING DATA")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print(f"Training set: {len(X_train)} samples")
print(f"Testing set: {len(X_test)} samples")

# STEP 3: Train the k-Nearest Neighbors classifier
print("\n3. TRAINING MODEL")
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
print("Model trained (Using k=5 nearest neighbors)")

# STEP 4: Make predictions and evaluate
print("\n4. EVALUATING MODEL")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) * 100
print(f"Accuracy: {accuracy:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Show confusion matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Visualize results
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Plot 1: Confusion Matrix
axes[0].imshow(cm, cmap='Blues')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')
axes[0].set_title('Confusion Matrix')
for i in range(len(cm)):
    for j in range(len(cm)):
        axes[0].text(j, i, str(cm[i, j]), ha='center', va='center', color='white')
axes[0].set_xticks(range(3))
axes[0].set_yticks(range(3))
axes[0].set_xticklabels(iris.target_names)
axes[0].set_yticklabels(iris.target_names)

# Plot 2: Feature comparison
feature_means = df.groupby('species_name')[iris.feature_names].mean()
feature_means.plot(kind='bar', ax=axes[1])
axes[1].set_title('Average Feature Values by Species')
axes[1].set_ylabel('Measurement (cm)')
axes[1].legend(title='Features', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
axes[1].tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.show()

# STEP 5: Test the model with new samples
print("\n5. TESTING WITH NEW SAMPLES")
new_sample = np.array([[5.0, 3.5, 1.5, 0.2]])
prediction = model.predict(new_sample)
confidence = model.predict_proba(new_sample)

print(f"Input: {new_sample[0]}")
print(f"Predicted Species: {iris.target_names[prediction[0]]}")
print(f"Confidence: {confidence[0][prediction[0]]:.2%}")
