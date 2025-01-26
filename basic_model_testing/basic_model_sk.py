from sklearn.model_selection import train_test_split
import numpy as np

# Simulated data
images = np.arange(1000).reshape(-1, 1)  # 1000 samples, 1 feature each
labels = np.arange(1000)  # Corresponding labels

# Proportions
test_size = 0.2
val_size = 0.1

# First split: Train and Temp (Temp will be split into Validation and Test)
X_train, X_temp, y_train, y_temp = train_test_split(
    images, labels, 
    test_size=test_size + val_size, 
    random_state=42
)

# Second split: Temp into Validation and Test
val_fraction = val_size / (test_size + val_size)  # Proportion of temp for validation (so the split would be test_size=0.2 and val_size=0.1, then test_size + val_size = 0.3 )

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, 
    test_size=val_fraction, #alr determined to be 33.33%
    random_state=42
)

# Print sizes
print(f"Training set size: {len(X_train)}")
print(f"Validation set size: {len(X_val)}")
print(f"Test set size: {len(X_test)}")
