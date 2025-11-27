import numpy as np
import glob
import cv2
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

img_size = 128  # Resize size

def load_images_from_folder(folder, label):
    X, y = [], []
    for img_path in glob.glob(folder + "/*.jpg"):
        img = cv2.imread(img_path)
        if img is None:
            continue

        img = cv2.resize(img, (img_size, img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img / 255.0

        # HOG feature extraction (optimized)
        features = hog(img,
                       orientations=9,
                       pixels_per_cell=(16, 16),
                       cells_per_block=(2, 2),
                       block_norm='L2-Hys',
                       transform_sqrt=True)

        X.append(features)
        y.append(label)
    return X, y


# Load training dataset
X_cat, y_cat = load_images_from_folder("train/cat", 0)
X_dog, y_dog = load_images_from_folder("train/dog", 1)

print("Cats loaded:", len(X_cat))
print("Dogs loaded:", len(X_dog))

X = np.array(X_cat + X_dog)
y = np.array(y_cat + y_dog)


# Split train/validation with stratification
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Use SVM (better than MLP here)
model = SVC(kernel='rbf', gamma='scale', C=5)
model.fit(X_train, y_train)

# Validate
prediction = model.predict(X_val)
acc = accuracy_score(y_val, prediction)
print("Validation Accuracy:", acc)


# Test
X_test_cat, y_test_cat = load_images_from_folder("test/cat", 0)
X_test_dog, y_test_dog = load_images_from_folder("test/dog", 1)

X_test = np.array(X_test_cat + X_test_dog)
y_test = np.array(y_test_cat + y_test_dog)

prediction_test = model.predict(X_test)
test_acc = accuracy_score(y_test, prediction_test)
print("Test Accuracy:", test_acc)


# Save model
joblib.dump(model, "cat_dog_model.pkl")
print("Model saved!")


# Predict on new images
def predict_image(path, model):
    img = cv2.imread(path)
    if img is None:
        print(path, "--> Cannot read image")
        return

    img = cv2.resize(img, (img_size, img_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img / 255.0

    features = hog(img,
                   orientations=9,
                   pixels_per_cell=(16, 16),
                   cells_per_block=(2, 2),
                   block_norm='L2-Hys',
                   transform_sqrt=True)

    pred = model.predict([features])[0]
    label = "dog" if pred == 1 else "cat"
    print(path, "-->", label)


for img in glob.glob("internet/*.*"):
    predict_image(img, model)
