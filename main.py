import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.layers import Dense, Flatten, Dropout, BatchNormalization
from keras.models import Sequential
from keras.datasets import mnist
from keras.optimizers import Adam
from sklearn.metrics import precision_score, recall_score, roc_curve, auc, accuracy_score
from sklearn.preprocessing import LabelBinarizer

# Wczytaj dane mnist_train i mnist_test
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalizacja wartości pikseli od 0 do 1
train_images = train_images / 255.0
test_images = test_images / 255.0

# Wypisz rozmiary zbiorów treningowego i testowego
print("Trening:", train_images.shape)
print("Test:", test_images.shape)

plt.imshow(train_images[1], cmap='binary')
plt.axis('off')
plt.show()
print(train_labels[1])

train_labels = keras.utils.to_categorical(train_labels, 10)
test_labels = keras.utils.to_categorical(test_labels, 10)

print(train_labels[1])

model = Sequential()

model.add(Flatten(input_shape=(28, 28)))  # Dla warstwy wejściowej

model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(10, activation='softmax'))

# Współczynnik
custom_optimizer = Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy',
              optimizer=custom_optimizer,
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, batch_size=32, epochs=20, validation_split=0.2)

# Ewaluacja modelu na danych testowych
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Dokładność klasyfikacji na danych testowych: {test_acc}')

# Wyświetlenie krzywej uczenia
acc = history.history['accuracy']
loss = history.history['loss']

epochs = range(1, len(acc) + 1)

plt.figure(figsize=(12, 6))  # Увеличение размера графика

plt.subplot(1, 2, 1)
plt.plot(epochs, acc, 'b-', label='Dokładność treningowa')  # Представление данных в виде линии
plt.title('Dokładność treningowa')
plt.xlabel('Epoki')
plt.ylabel('Dokładność')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, loss, 'b-', label='Strata treningowa')  # Представление данных в виде линии
plt.title('Strata treningowa')
plt.xlabel('Epoki')
plt.ylabel('Strata')
plt.legend()

plt.tight_layout()
plt.show()



# Predykcje na zbiorze testowym
predictions = model.predict(test_images)

# Konwersja wyników na etykiety klas
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(test_labels, axis=1)

# Obliczenie precyzji
precision = precision_score(true_labels, predicted_labels, average='weighted')
print(f'Precyzja: {precision}')

# Obliczenie czułości
recall = recall_score(true_labels, predicted_labels, average='weighted')
print(f'Czułość: {recall}')

# Obliczenie krzywej ROC
lb = LabelBinarizer()
true_labels_bin = lb.fit_transform(true_labels)
predicted_labels_bin = lb.transform(predicted_labels)

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(10):  # Zakładając 10 klas
    fpr[i], tpr[i], _ = roc_curve(true_labels_bin[:, i], predicted_labels_bin[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Wykres krzywej ROC dla każdej klasy
plt.figure(figsize=(8, 6))
for i in range(10):
    plt.plot(fpr[i], tpr[i], label='ROC curve (class {})'.format(i))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()

for i in range(10):
    fpr_class = fpr[i]
    tpr_class = tpr[i]
    roc_auc_class = roc_auc[i]

    print(f"Klasa {i}:")
    print(f"FPR: {fpr_class}")
    print(f"TPR: {tpr_class}")
    print(f"AUC: {roc_auc_class}")
    print("\n")

k = 6
plt.imshow(test_images[k], cmap='binary')
plt.axis('off')
plt.show()
print(test_labels[k])

k = 10
plt.imshow(test_images[k], cmap='binary')
plt.axis('off')
plt.show()
print(test_labels[k])


