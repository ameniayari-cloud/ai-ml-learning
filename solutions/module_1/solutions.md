# Solutions - Module 1

1. `load_iris(as_frame=True)` puis `iris.frame.head()`.
2. `df.describe()` ou `df.agg(['mean','std'])`.
3. `train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)`.
4. Pour éviter la fuite de données: `scaler.fit(X_train)` puis `transform` sur train/test.
