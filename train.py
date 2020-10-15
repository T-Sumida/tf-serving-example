# -*- coding:utf-8 -*-
import os
import yaml
import argparse
import numpy as np
import tensorflow as tf
from typing import Tuple

from model import SimpleCNN


def parse_args() -> argparse.Namespace:
    """引数解析

    Returns:
        argparse.Namespace: 引数情報
    """
    parser = argparse.ArgumentParser(
        prog="train.py",
        description="tensorflow-serving training script",
        add_help=True
    )
    parser.add_argument(
        "config_file", type=str,
        help="training config yaml file"
    )
    return parser.parse_args()


def load_dataset() -> Tuple[Tuple[np.array, np.array], Tuple[np.array, np.array]]:
    """データセットをロード

    Returns:
        Tuple[Tuple[np.array, np.array], Tuple[np.array, np.array]]: trainデータとtestデータ
    """
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    train_images = train_images / 255.0
    test_images = test_images / 255.0
    return (train_images, train_labels), (test_images, test_labels)


def main():
    args = parse_args()

    with open(args.config_file, 'r') as f:
        cfg = yaml.load(f)

    # load dataset
    (train_imgs, train_labels), (test_imgs, test_labels) = load_dataset()
    print("Train image shape: ", train_imgs.shape)
    print("Test image shape: ", test_imgs.shape)

    # train
    cnn_model = SimpleCNN(cfg['global']['tmp_path'])
    cnn_model.train(cfg['global']['epoch'], train_imgs, train_labels, test_imgs, test_labels)

    # savedModelに書き出し
    cnn_model.export_SavedModel(
        cfg['tf-serving']['version_number'],
        cfg['tf-serving']['export_dir'],
        weight_path=os.path.join(cfg['global']['tmp_path'], "model.h5")
    )


if __name__ == "__main__":
    main()