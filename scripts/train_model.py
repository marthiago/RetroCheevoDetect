import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os

# Diretórios
train_images_dir = '/Users/thiagomartins/Projetos/RetroCheevoDetect/data/train/images/duckstation_notification'
validation_images_dir = '/Users/thiagomartins/Projetos/RetroCheevoDetect/data/validation/images/duckstation_notification'

# Verificação do número de imagens
num_train_images = len(os.listdir(train_images_dir))
num_validation_images = len(os.listdir(validation_images_dir))
batch_size = 20

print(f"Number of training images: {num_train_images}")
print(f"Number of validation images: {num_validation_images}")

# Configuração dos Geradores de Dados
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    directory=os.path.dirname(train_images_dir),
    target_size=(150, 150),
    batch_size=batch_size,
    class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
    directory=os.path.dirname(validation_images_dir),
    target_size=(150, 150),
    batch_size=batch_size,
    class_mode='binary'
)

# Função para repetir o gerador indefinidamente
def infinite_generator(generator):
    while True:
        for batch in generator:
            yield batch

# Definição do Modelo
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(150, 150, 3)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-4),
    metrics=['accuracy']
)

# Ajuste dos steps_per_epoch e validation_steps
steps_per_epoch = max(1, num_train_images // batch_size)
validation_steps = max(1, num_validation_images // batch_size)

print(f"Steps per epoch: {steps_per_epoch}")
print(f"Validation steps: {validation_steps}")

history = model.fit(
    infinite_generator(train_generator),
    steps_per_epoch=steps_per_epoch,
    epochs=120,
    validation_data=infinite_generator(validation_generator),
    validation_steps=validation_steps
)

# Salvar o modelo no formato .keras
model.save('/Users/thiagomartins/Projetos/RetroCheevoDetect/models/retroachievements_model.keras')

# Plotar as métricas de treinamento e validação
plt.figure(figsize=(12, 4))

# Perda
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Precisão
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
