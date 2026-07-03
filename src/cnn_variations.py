"""Variações de arquitetura CNN usadas nos experimentos do projeto.

Este arquivo centraliza as topologias de Redes Neurais Convolucionais (CNNs)
citadas no relatório: modelo base, BatchNormalization, AumentaProfundidade,
ArquiteturaRasa e GlobalAveragePooling.

Observação:
    As versões originais foram testadas trocando manualmente trechos de código
    durante os experimentos. Portanto, este arquivo reconstrói as variações de
    forma compatível com o código atual do repositório, que usa imagens 64x64x3.
"""

from __future__ import annotations

from typing import Callable, Dict, Tuple

from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling2D,
    MaxPooling2D,
)
from tensorflow.keras.models import Sequential


InputShape = Tuple[int, int, int]
ModelBuilder = Callable[[InputShape, int], Sequential]


DEFAULT_INPUT_SHAPE: InputShape = (64, 64, 3)
DEFAULT_NUM_CLASSES = 8


def _compile(model: Sequential) -> Sequential:
    """Compila o modelo com a mesma configuração usada no projeto."""
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_cnn_base(
    input_shape: InputShape = DEFAULT_INPUT_SHAPE,
    num_classes: int = DEFAULT_NUM_CLASSES,
) -> Sequential:
    """CNN base.

    Arquitetura convolucional inicial de referência, sem BatchNormalization e
    sem GlobalAveragePooling.
    """
    model = Sequential(name="cnn_base")
    model.add(Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), activation="relu", padding="same"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation="softmax"))
    return _compile(model)


def build_cnn_batchnormalization(
    input_shape: InputShape = DEFAULT_INPUT_SHAPE,
    num_classes: int = DEFAULT_NUM_CLASSES,
) -> Sequential:
    """CNN BatchNormalization.

    Mantém uma arquitetura próxima da CNN base, mas adiciona BatchNormalization
    após as camadas convolucionais e na camada densa.
    """
    model = Sequential(name="cnn_batchnormalization")
    model.add(Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), activation="relu", padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation="softmax"))
    return _compile(model)


def build_cnn_aumenta_profundidade(
    input_shape: InputShape = DEFAULT_INPUT_SHAPE,
    num_classes: int = DEFAULT_NUM_CLASSES,
) -> Sequential:
    """CNN AumentaProfundidade.

    Esta é a variação mais próxima do arquivo CNNcorrigido.py atual. O script
    existente imprime "CNN AumentaProfundidade" e usa quatro blocos
    convolucionais: 32 -> 64 -> 128 -> 128.

    Observação: o código atual também usa BatchNormalization e
    GlobalAveragePooling2D. Por isso, esta reconstrução preserva esses elementos
    para ficar compatível com o que está salvo no GitHub.
    """
    model = Sequential(name="cnn_aumenta_profundidade")
    model.add(Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), activation="relu", padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), activation="relu", padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(GlobalAveragePooling2D())
    model.add(Dense(256, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation="softmax"))
    return _compile(model)


def build_cnn_arquitetura_rasa(
    input_shape: InputShape = DEFAULT_INPUT_SHAPE,
    num_classes: int = DEFAULT_NUM_CLASSES,
) -> Sequential:
    """CNN ArquiteturaRasa.

    Usa menos camadas convolucionais para avaliar uma versão mais simples e
    mais rápida da CNN.
    """
    model = Sequential(name="cnn_arquitetura_rasa")
    model.add(Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation="softmax"))
    return _compile(model)


def build_cnn_global_average_pooling(
    input_shape: InputShape = DEFAULT_INPUT_SHAPE,
    num_classes: int = DEFAULT_NUM_CLASSES,
) -> Sequential:
    """CNN GlobalAveragePooling.

    Substitui o Flatten por GlobalAveragePooling2D para reduzir a quantidade de
    parâmetros e ajudar a controlar overfitting.
    """
    model = Sequential(name="cnn_global_average_pooling")
    model.add(Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), activation="relu", padding="same"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(GlobalAveragePooling2D())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation="softmax"))
    return _compile(model)


CNN_VARIATIONS: Dict[str, ModelBuilder] = {
    "cnn_base": build_cnn_base,
    "cnn_batchnormalization": build_cnn_batchnormalization,
    "cnn_aumenta_profundidade": build_cnn_aumenta_profundidade,
    "cnn_arquitetura_rasa": build_cnn_arquitetura_rasa,
    "cnn_global_average_pooling": build_cnn_global_average_pooling,
}
