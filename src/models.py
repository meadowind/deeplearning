from tensorflow.keras import layers, models, applications

def get_iteration_1_baseline(num_classes):
    """Simple CNN to establish a baseline."""
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def get_iteration_2_transfer(num_classes):
    """MobileNetV2: Solving accuracy issues with Transfer Learning."""
    base = applications.MobileNetV2(input_shape=(32,32,3), include_top=False, weights='imagenet')
    base.trainable = False 
    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3), # Improvement: Regularization to stop overfitting
        layers.Dense(num_classes, activation='softmax')
    ])
    return model