#Main script to run training

from src.preprocess import load_dataset
from src.models import get_iteration_1_baseline
import matplotlib.pyplot as plt

# 1. Load Data
X_train, y_train = load_dataset('data/Training')

# 2. Build and Train
model = get_iteration_1_baseline(62)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("Starting training for Iteration 1...")
history = model.fit(X_train, y_train, epochs=10, validation_split=0.2)

# 3. Save Evidence (Rubric: Model Selection)
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.title('Iteration 1: Baseline Learning Curve')
plt.savefig('notebooks/iteration_1_curve.png')
model.save('models/baseline.h5')
