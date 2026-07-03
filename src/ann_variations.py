"""Variações de arquitetura ANN usadas nos experimentos do projeto.

Este arquivo centraliza as topologias de Redes Neurais Artificiais (ANNs)
citadas no relatório: modelo base, ReduzComplexidade, RedeMaisProfunda,
BatchNormalization e FunilInvertido.

Observação:
    As versões originais foram testadas trocando manualmente trechos de código
    durante os experimentos. Portanto, este arquivo reconstrói as variações de
    forma compatível com o código atual do repositório, que usa imagens 64x64
    achatadas em vetores de 64 * 64 * 3.
"""

from __future__ import annotations

from typing import Callable, Dict, Tuple

from tensorflow.keras.layers import BatchNormalization, Dense, Dropout
from tensorflow.keras.models import Sequential


InputShape = Tuple[int, ...]
ModelBuilder = Callable[[InputShape, int], Sequential]


DEFAULT_INPUT_SHAPE: InputShape = (64 * 64 * 3,)
DEFAULT_NUM_CLASSES = 8


def _compile(model: Sequential) -> Sequential:
    """Compila o modelo com a mesma configuração usada no projeto."""
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_ann_base(
    input_shape: InputShape = DEFAULT_INPUT_SHAPE,
    num_classes: int = DEFAULT_NUM_CLASSES,
) -> Sequential:
    """ANN base.

    Arquitetura de referência com camadas densas em ordem decrescente.
    Esta versão não adiciona BatchNormalization, porque ela serve como base
    para comparação com a variação específica de normalização.
    """
    model = Sequential(name="ann_base")
    model.add(Dense(256, activation="relu", input_shape=input_shape))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation="softmax"))
    return _compile(model)


def build_ann_reduz_complexidade(
    input_shape: InputShape = DEFAULT_INPUT_SHAPE,
    num_classes: int = DEFAULT_NUM_CLASSES,
) -> Sequential:
    """ANN ReduzComplexidade.

    Usa menos neurônios por camada para reduzir o custo computacional.
    """
    model = Sequential(name="ann_reduz_complexidade")
    model.add(Dense(128, activation="relu", input_shape=input_shape))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation="softmax"))
    return _compile(model)


def build_ann_rede_mais_profunda(
    input_shape: InputShape = DEFAULT_INPUT_SHAPE,
    num_classes: int = DEFAULT_NUM_CLASSES,
) -> Sequential:
    """ANN RedeMaisProfunda.

    Adiciona mais camadas ocultas para testar se maior profundidade melhora
    a capacidade de aprendizagem da ANN.
    """
    model = Sequential(name="ann_rede_mais_profunda")
    model.add(Dense(512, activation="relu", input_shape=input_shape))
    model.add(Dropout(0.3))
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation="softmax"))
    return _compile(model)


def build_ann_batchnormalization(
    input_shape: InputShape = DEFAULT_INPUT_SHAPE,
    num_classes: int = DEFAULT_NUM_CLASSES,
) -> Sequential:
    """ANN BatchNormalization.

    Esta é a variação que já está aplicada no arquivo ANNcorrigido.py atual:
    camadas Dense 256 -> 128 -> 64 -> 32, cada uma seguida de
    BatchNormalization e Dropout.
    """
    model = Sequential(name="ann_batchnormalization")
    model.add(Dense(256, activation="relu", input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(128, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(64, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(32, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation="softmax"))
    return _compile(model)


def build_ann_funil_invertido(
    input_shape: InputShape = DEFAULT_INPUT_SHAPE,
    num_classes: int = DEFAULT_NUM_CLASSES,
) -> Sequential:
    """ANN FunilInvertido.

    Testa a ordem invertida das camadas, começando com menos neurônios e
    aumentando a largura das camadas ao longo da rede.
    """
    model = Sequential(name="ann_funil_invertido")
    model.add(Dense(32, activation="relu", input_shape=input_shape))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation="softmax"))
    return _compile(model)


ANN_VARIATIONS: Dict[str, ModelBuilder] = {
    "ann_base": build_ann_base,
    "ann_reduz_complexidade": build_ann_reduz_complexidade,
    "ann_rede_mais_profunda": build_ann_rede_mais_profunda,
    "ann_batchnormalization": build_ann_batchnormalization,
    "ann_funil_invertido": build_ann_funil_invertido,
}
