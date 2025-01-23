import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

class BinaryImageClassifier:
    def __init__(self, peak_dir, non_peak_dir, image_size=(300, 300), batch_size=64, epochs=10, patience=3):
        self.peak_dir = peak_dir
        self.non_peak_dir = non_peak_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.model = None

    def load_and_prepare_data(self):
        images = []
        labels = []

        # Load peak images
        for image_file in os.listdir(self.peak_dir):
            image = load_img(os.path.join(self.peak_dir, image_file), color_mode='grayscale', target_size=self.image_size)
            images.append(img_to_array(image))
            labels.append(1)

        # Load non-peak images
        for image_file in os.listdir(self.non_peak_dir):
            image = load_img(os.path.join(self.non_peak_dir, image_file), color_mode='grayscale', target_size=self.image_size)
            images.append(img_to_array(image))
            labels.append(0)

        images = np.array(images) / 255.0  # Normalize images
        labels = np.array(labels)
        return train_test_split(images, labels, test_size=0.2, random_state=42)

    def build_model(self):
        model = Sequential()

        # First conv block
        model.add(Conv2D(32, (3, 3), padding='same', input_shape=(self.image_size[0], self.image_size[1], 1)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # Second conv block
        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # Third conv block
        model.add(Conv2D(128, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # Fourth conv block
        model.add(Conv2D(256, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # Fully connected layers
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
        self.model = model

    def train_model(self, x_train, y_train, x_val, y_val):
        early_stopping = EarlyStopping(monitor='val_loss', patience=self.patience)
        self.model.fit(x_train, y_train, validation_data=(x_val, y_val), 
                       epochs=self.epochs, batch_size=self.batch_size, callbacks=[early_stopping])

    def evaluate_model(self, x_val, y_val):
        score = self.model.evaluate(x_val, y_val, verbose=0)
        print(f'Test loss: {score[0]}')
        print(f'Test accuracy: {score[1]}')

# Example usage
if __name__ == "__main__":
    
    folder_path = os.path.dirname(os.path.abspath(__file__))
    
    PeakSpectrogram = os.path.join(folder_path, "PeakspectrogramType1")
    
    NonPeakSpectrogram = os.path.join(folder_path, "NonPeakspectrogram")

    classifier = BinaryImageClassifier(PeakSpectrogram, NonPeakSpectrogram)
    x_train, x_val, y_train, y_val = classifier.load_and_prepare_data()
    classifier.build_model()
    classifier.train_model(x_train, y_train, x_val, y_val)
    classifier.evaluate_model(x_val, y_val)
