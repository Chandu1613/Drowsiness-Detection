# Basic libraries
import numpy as np

# CNN libraries
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# Metric Libraries
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

# ignore warnings
import warnings
warnings.filterwarnings('ignore')


train_dir = 'dataset/train'
test_dir = 'dataset/test'

train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
                    train_dir,
                    target_size = (224, 224),
                    batch_size = 32,
                    class_mode = 'categorical',
                    shuffle = True,
                    seed = 42,
                    subset='training')

val_generator = train_datagen.flow_from_directory(
                    train_dir,
                    target_size = (224, 224),
                    batch_size = 32,
                    class_mode = 'categorical',
                    shuffle = True,
                    seed = 42,
                    subset='validation')

test_generator = test_datagen.flow_from_directory(
                    test_dir,
                    target_size = (224, 224),
                    batch_size = 32,
                    class_mode = 'categorical',
                    shuffle = False)

model = Sequential(
    [
        Conv2D(16,kernel_size=3,activation='relu',input_shape=(224,224,3)),
        MaxPooling2D((2,2)),

        Conv2D(32,kernel_size=3,activation='relu',padding='same'),
        MaxPooling2D((2,2)),
        BatchNormalization(),
        Dropout(0.3),

        Conv2D(64,kernel_size=3,activation='relu',padding='same'),
        MaxPooling2D((2,2)),
        BatchNormalization(),
        Dropout(0.3),

        Conv2D(128,kernel_size=3,activation='relu',padding='same'),
        MaxPooling2D((2,2)),
        BatchNormalization(),
        Dropout(0.3),

        Flatten(),

        Dense(128,activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        Dense(4,activation='softmax')
    ]
)

model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()


early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, mode='max', verbose=1, restore_best_weights=True)

model_checkpoint = ModelCheckpoint(filepath='best_model.keras',  #  Change .h5 → .keras
                                   monitor='val_accuracy',
                                   save_best_only=True,
                                   save_weights_only=False,  # Change to False to save full model
                                   mode='max',
                                   verbose=1)
history = model.fit(
    train_generator, 
    epochs=30,
    validation_data=val_generator,
    callbacks=[early_stopping, model_checkpoint]
)

test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc:.2f}")
print(f"Test Loss: {test_loss:.2f}")

y_pred = model.predict(test_generator)
y_pred_class = np.argmax(y_pred, axis=1)
true_class = test_generator.classes

# metrics
cm = ConfusionMatrixDisplay.from_predictions(true_class, y_pred_class, display_labels=test_generator.class_indices)
print('Classification Report:\n\n', classification_report(true_class, y_pred_class))