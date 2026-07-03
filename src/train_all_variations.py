"""Treina todas as variações ANN e CNN descritas no relatório.

Uso básico, dentro da raiz do repositório:

    python src/train_all_variations.py --family all

Pré-requisito:
    O arquivo captured_data/mask_data.csv deve existir e conter as colunas:
    image_path,label

Saídas geradas:
    results/figures/               gráficos de acurácia e loss
    results/confusion_matrices/    matrizes de confusão
    results/tables/metricas_variacoes.csv
    models/                        modelos .keras, quando --save-models for usado
"""

from __future__ import annotations

import argparse
import csv
import os
import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split

from ann_variations import ANN_VARIATIONS
from cnn_variations import CNN_VARIATIONS


LABEL_MAP = {
    0: "Sem nada",
    1: "Capacete",
    2: "Óculos",
    3: "Máscara",
    4: "Capacete e Óculos",
    5: "Capacete e Máscara",
    6: "Máscara e Óculos",
    7: "Capacete, Óculos e Máscara",
}


def set_seed(seed: int) -> None:
    """Define sementes para deixar os testes mais reprodutíveis."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def read_dataset(csv_path: Path, image_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Lê as imagens indicadas no CSV e retorna X e y.

    O CSV deve ter as colunas image_path e label, como já é gerado nos scripts
    ANNcorrigido.py e CNNcorrigido.py.
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV não encontrado: {csv_path}. "
            "Capture/organize as imagens primeiro ou ajuste --csv."
        )

    data = pd.read_csv(csv_path)
    required_columns = {"image_path", "label"}
    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        raise ValueError(f"O CSV precisa conter as colunas: {sorted(required_columns)}")

    images: List[np.ndarray] = []
    labels: List[int] = []
    missing_files: List[str] = []

    for _, row in data.iterrows():
        image_path = Path(str(row["image_path"]))
        label = int(row["label"])

        img = cv2.imread(str(image_path))
        if img is None:
            missing_files.append(str(image_path))
            continue

        img = cv2.resize(img, (image_size, image_size))
        images.append(img)
        labels.append(label)

    if not images:
        raise ValueError("Nenhuma imagem válida foi carregada. Confira os caminhos do CSV.")

    if missing_files:
        print(f"Aviso: {len(missing_files)} imagem(ns) não foram encontradas e foram ignoradas.")
        for item in missing_files[:10]:
            print(f"  - {item}")
        if len(missing_files) > 10:
            print("  ...")

    x = np.array(images, dtype="float32") / 255.0
    y = np.array(labels, dtype="int64")
    return x, y


def can_stratify(labels: np.ndarray) -> bool:
    """Verifica se é seguro usar stratify no train_test_split."""
    _, counts = np.unique(labels, return_counts=True)
    return bool(np.all(counts >= 2))


def plot_training_curves(history, output_prefix: Path) -> None:
    """Salva os gráficos de acurácia e loss."""
    # Loss
    plt.figure()
    plt.plot(history.history.get("loss", []), label="Loss Treino")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="Loss Validação")
    plt.xlabel("Épocas")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(f"Loss - {output_prefix.name}")
    plt.tight_layout()
    plt.savefig(output_prefix.with_name(f"{output_prefix.name}_loss.png"), dpi=150)
    plt.close()

    # Accuracy
    plt.figure()
    plt.plot(history.history.get("accuracy", []), label="Acurácia Treino")
    if "val_accuracy" in history.history:
        plt.plot(history.history["val_accuracy"], label="Acurácia Validação")
    plt.xlabel("Épocas")
    plt.ylabel("Acurácia")
    plt.legend()
    plt.title(f"Acurácia - {output_prefix.name}")
    plt.tight_layout()
    plt.savefig(output_prefix.with_name(f"{output_prefix.name}_accuracy.png"), dpi=150)
    plt.close()


def plot_confusion_matrix(model, x_val: np.ndarray, y_val: np.ndarray, output_path: Path) -> None:
    """Salva a matriz de confusão do conjunto de validação."""
    y_pred_probs = model.predict(x_val, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    labels_present = sorted(set(y_val.tolist()) | set(y_pred.tolist()))
    display_labels = [LABEL_MAP.get(label, str(label)) for label in labels_present]

    cm = confusion_matrix(y_val, y_pred, labels=labels_present)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)

    plt.figure(figsize=(10, 8))
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45, values_format="d")
    plt.title(output_path.stem.replace("_", " ").title())
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def write_metrics_csv(rows: List[Dict[str, object]], output_path: Path) -> None:
    """Salva as métricas finais de cada variação em CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "familia",
        "variacao",
        "loss_treino_final",
        "acuracia_treino_final",
        "loss_validacao_final",
        "acuracia_validacao_final",
        "epocas",
        "batch_size",
        "image_size",
        "test_size",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def select_variations(family: str, only: str | None) -> Dict[str, object]:
    """Seleciona quais variações serão treinadas."""
    variations: Dict[str, object] = {}

    if family in {"all", "ann"}:
        variations.update(ANN_VARIATIONS)
    if family in {"all", "cnn"}:
        variations.update(CNN_VARIATIONS)

    if only:
        requested = [item.strip() for item in only.split(",") if item.strip()]
        unknown = [item for item in requested if item not in variations]
        if unknown:
            available = ", ".join(sorted(variations))
            raise ValueError(f"Variação desconhecida: {unknown}. Disponíveis: {available}")
        variations = {name: variations[name] for name in requested}

    return variations


def train_variations(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    csv_path = Path(args.csv)
    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"
    confusion_dir = output_dir / "confusion_matrices"
    tables_dir = output_dir / "tables"
    models_dir = Path(args.models_dir)

    figures_dir.mkdir(parents=True, exist_ok=True)
    confusion_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    if args.save_models:
        models_dir.mkdir(parents=True, exist_ok=True)

    x_images, y = read_dataset(csv_path, args.image_size)
    num_classes = max(8, int(y.max()) + 1)

    stratify = y if can_stratify(y) else None
    if stratify is None:
        print("Aviso: não foi possível usar stratify porque alguma classe possui menos de 2 imagens.")

    x_train_img, x_val_img, y_train, y_val = train_test_split(
        x_images,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=stratify,
    )

    variations = select_variations(args.family, args.only)
    metrics_rows: List[Dict[str, object]] = []

    for variation_name, builder in variations.items():
        print("=" * 80)
        print(f"Treinando: {variation_name}")

        is_ann = variation_name.startswith("ann_")
        family = "ANN" if is_ann else "CNN"

        if is_ann:
            x_train = x_train_img.reshape((x_train_img.shape[0], -1))
            x_val = x_val_img.reshape((x_val_img.shape[0], -1))
            input_shape = (args.image_size * args.image_size * 3,)
        else:
            x_train = x_train_img
            x_val = x_val_img
            input_shape = (args.image_size, args.image_size, 3)

        model = builder(input_shape=input_shape, num_classes=num_classes)
        model.summary()

        history = model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=args.epochs,
            batch_size=args.batch_size,
            verbose=1,
        )

        prefix = figures_dir / variation_name
        plot_training_curves(history, prefix)
        plot_confusion_matrix(
            model,
            x_val,
            y_val,
            confusion_dir / f"{variation_name}_confusion_matrix.png",
        )

        if args.save_models:
            model.save(models_dir / f"{variation_name}.keras")

        metrics_rows.append(
            {
                "familia": family,
                "variacao": variation_name,
                "loss_treino_final": history.history["loss"][-1],
                "acuracia_treino_final": history.history["accuracy"][-1],
                "loss_validacao_final": history.history["val_loss"][-1],
                "acuracia_validacao_final": history.history["val_accuracy"][-1],
                "epocas": args.epochs,
                "batch_size": args.batch_size,
                "image_size": args.image_size,
                "test_size": args.test_size,
            }
        )

    write_metrics_csv(metrics_rows, tables_dir / "metricas_variacoes.csv")
    print("=" * 80)
    print("Treinamento concluído.")
    print(f"Gráficos: {figures_dir}")
    print(f"Matrizes de confusão: {confusion_dir}")
    print(f"Métricas: {tables_dir / 'metricas_variacoes.csv'}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Treina as variações ANN/CNN descritas no relatório do projeto."
    )
    parser.add_argument(
        "--csv",
        default="captured_data/mask_data.csv",
        help="Caminho para o CSV com colunas image_path,label.",
    )
    parser.add_argument(
        "--family",
        choices=["all", "ann", "cnn"],
        default="all",
        help="Família de modelos a treinar.",
    )
    parser.add_argument(
        "--only",
        default=None,
        help=(
            "Lista separada por vírgula com variações específicas. Exemplo: "
            "--only ann_batchnormalization,cnn_aumenta_profundidade"
        ),
    )
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--save-models", action="store_true")
    return parser


if __name__ == "__main__":
    train_variations(build_arg_parser().parse_args())
