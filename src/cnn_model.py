# CNNReconhecimentoObjetos_v2_testeExterno.py
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout, GlobalAveragePooling2D, Dense
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os

data_dir = "captured_data"
os.makedirs(data_dir, exist_ok=True)
csv_path = os.path.join(data_dir, "mask_data.csv")
if not os.path.exists(csv_path):
    df = pd.DataFrame(columns=["image_path", "label"])
    df.to_csv(csv_path, index=False)

def save_to_csv(image_path, label):
    df = pd.read_csv(csv_path)
    df = df._append({"image_path": image_path, "label": label}, ignore_index=True)
    df.to_csv(csv_path, index=False)

camera = cv2.VideoCapture(1)
print("Pressione 'c' para salvar uma imagem com Capacete.")
print("Pressione 'o' para salvar uma imagem com Óculos.")
print("Pressione 'm' para salvar uma imagem com Máscara.")
print("Pressione 'q' para salvar uma imagem com Capacete e Óculos.")
print("Pressione 'w' para salvar uma imagem com Capacete e Máscara.")
print("Pressione 'y' para salvar uma imagem com Máscara e Óculos.")
print("Pressione 'z' para salvar uma imagem com Capacete, Óculos e Máscara.")
print("Pressione 'n' para salvar uma imagem Sem Nada.")
print("Pressione 't' para treinar todo o dataset.")
print("Pressione 'x' para treinar e testar com train_test_split a 70%.")
print("Pressione 'v' para validar em tempo real.")
print("Pressione 'e' para sair.")

while True:
    ret, frame = camera.read()
    if not ret:
        print("Erro ao acessar a câmera.")
        break

    cv2.imshow("Captura de Imagem", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        image_path = os.path.join(data_dir, f"cap_{int(cv2.getTickCount())}.jpg")
        cv2.imwrite(image_path, frame)
        save_to_csv(image_path, 1)
        print(f"Imagem com Capacete salva: {image_path}")

    elif key == ord('o'):
        image_path = os.path.join(data_dir, f"oc_{int(cv2.getTickCount())}.jpg")
        cv2.imwrite(image_path, frame)
        save_to_csv(image_path, 2)
        print(f"Imagem com Óculos salva: {image_path}")

    elif key == ord('m'):
        image_path = os.path.join(data_dir, f"mask_{int(cv2.getTickCount())}.jpg")
        cv2.imwrite(image_path, frame)
        save_to_csv(image_path, 3)
        print(f"Imagem com Máscara salva: {image_path}")

    elif key == ord('q'):
        image_path = os.path.join(data_dir, f"capoc_{int(cv2.getTickCount())}.jpg")
        cv2.imwrite(image_path, frame)
        save_to_csv(image_path, 4)
        print(f"Imagem com Capacete e Óculos salva: {image_path}")

    elif key == ord('w'):
        image_path = os.path.join(data_dir, f"capmask_{int(cv2.getTickCount())}.jpg")
        cv2.imwrite(image_path, frame)
        save_to_csv(image_path, 5)
        print(f"Imagem com Capacete e Máscara salva: {image_path}")

    elif key == ord('y'):
        image_path = os.path.join(data_dir, f"maskoc_{int(cv2.getTickCount())}.jpg")
        cv2.imwrite(image_path, frame)
        save_to_csv(image_path, 6)
        print(f"Imagem com Máscara e Óculos salva: {image_path}")

    elif key == ord('z'):
        image_path = os.path.join(data_dir, f"capocmask_{int(cv2.getTickCount())}.jpg")
        cv2.imwrite(image_path, frame)
        save_to_csv(image_path, 7)
        print(f"Imagem com Capacete, Óculos e Máscara salva: {image_path}")

    elif key == ord('n'):
        image_path = os.path.join(data_dir, f"nothing_{int(cv2.getTickCount())}.jpg")
        cv2.imwrite(image_path, frame)
        save_to_csv(image_path, 0)
        print(f"Imagem Sem Equipamento salva: {image_path}")

    elif key == ord('x'):
        data = pd.read_csv(csv_path)
        images = []
        labels = []
        for _, row in data.iterrows():
            img = cv2.imread(row['image_path'])
            img = cv2.resize(img, (64, 64))
            images.append(img)
            labels.append(row['label'])
        images = np.array(images) / 255.0
        labels = np.array(labels)
        from sklearn.model_selection import train_test_split
        x_train, x_val, y_train, y_val = train_test_split(images, labels, test_size=0.3, random_state=42)

        model = Sequential()
        model.add(Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(64, 64, 3)))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2,2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2,2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2,2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2,2)))
        model.add(Dropout(0.25))

        model.add(GlobalAveragePooling2D())

        model.add(Dense(256, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        model.add(Dense(8, activation='softmax'))
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        print("Treinando o modelo CNN AumentaProfundidade...")
        history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=70, batch_size=32)

        plt.plot(history.history['loss'], label='Loss Treino')
        plt.plot(history.history['val_loss'], label='Loss Validação')
        plt.xlabel('Épocas'); plt.ylabel('Loss'); plt.legend()
        plt.title('Loss durante o treino CNN Split'); plt.show()

        plt.plot(history.history['accuracy'], label='Acurácia Treino')
        plt.plot(history.history['val_accuracy'], label='Acurácia Validação')
        plt.xlabel('Épocas'); plt.ylabel('Acurácia'); plt.legend()
        plt.title('Acurácia durante o treino CNN Split'); plt.show()

        y_pred_probs = model.predict(x_val)
        y_pred = np.argmax(y_pred_probs, axis=1)
        cm = confusion_matrix(y_val, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        plt.title('Matriz de Confusão CNN Split'); plt.show()

        model.save('meuModeloCNN_v2.keras')
        print("Modelo CNN salvo com sucesso.")

    elif key == ord('t'):
        data = pd.read_csv(csv_path)
        images = []
        labels = []
        for _, row in data.iterrows():
            img = cv2.imread(row['image_path'])
            img = cv2.resize(img, (64, 64))
            images.append(img)
            labels.append(row['label'])
        images = np.array(images) / 255.0
        labels = np.array(labels)

        model = Sequential()
        model.add(Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(64, 64, 3)))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2,2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2,2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2,2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2,2)))
        model.add(Dropout(0.25))

        model.add(GlobalAveragePooling2D())

        model.add(Dense(256, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        model.add(Dense(8, activation='softmax'))
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        print("Treinando o modelo CNN completo...")
        history = model.fit(images, labels, epochs=70, batch_size=32)

        plt.plot(history.history['loss'], label='Loss Treino')
        plt.xlabel('Épocas'); plt.ylabel('Loss'); plt.legend()
        plt.title('Loss durante o treino CNN Conjunto Completo'); plt.show()

        plt.plot(history.history['accuracy'], label='Acurácia Treino')
        plt.xlabel('Épocas'); plt.ylabel('Acurácia'); plt.legend()
        plt.title('Acurácia durante o treino CNN Conjunto Completo'); plt.show()

        model.save('meuModeloCompletoCNN_v2.keras')
        print("Modelo CNN completo salvo com sucesso.")

    elif key == ord('v'):
        modelTest = tf.keras.models.load_model('meuModeloCompletoCNN_v2.keras')
        print("CNN carregado com sucesso.")
        img = cv2.resize(frame, (64, 64))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        prediction = modelTest.predict(img)
        predicted_class = np.argmax(prediction)
        label_map = {
            0: "Sem nada", 1: "Capacete", 2: "Óculos", 3: "Máscara",
            4: "Capacete e Óculos", 5: "Capacete e Máscara",
            6: "Máscara e Óculos", 7: "Capacete, Óculos e Máscara"
        }
        cv2.putText(frame, label_map[predicted_class], (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        print(f"Classe prevista pela CNN: {label_map[predicted_class]}")
        cv2.imshow("Validação CNN", frame)

    elif key == ord('e'):
        print("Finalizando a captura de imagens.")
        break

camera.release()
cv2.destroyAllWindows()
