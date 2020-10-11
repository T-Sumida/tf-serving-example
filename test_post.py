# -*- coding:utf-8 -*-
import json
import base64
import argparse
import requests
from io import BytesIO
from PIL import Image
from json import load
from typing import Tuple, List

import numpy as np
import tensorflow as tf


def parse_args() -> argparse.Namespace:
    """引数を解析する

    Returns:
        argparse.Namespace: 引数情報
    """
    parser = argparse.ArgumentParser(
        prog="test_post.py",
        description="tensorflow-serving test script",
        add_help=True
    )
    parser.add_argument(
        "--model_name", type=str, default="tf-serving",
        help="tf-serving model name"
    )
    parser.add_argument(
        "--version", type=str, default=None,
        help="tf-serving model version number"
    )
    return parser.parse_args()


def load_data() -> Tuple[np.array, np.array]:
    """テストデータをロードする

    Returns:
        Tuple[np.array, np.array]: 画像,ラベル
    """
    fashion_mnist = tf.keras.datasets.fashion_mnist
    _, (test_images, test_labels) = fashion_mnist.load_data()
    return (test_images, test_labels)


def convert_img2base64(imgs: np.array) -> List:
    """numpy画像をbase64に変換する

    Args:
        imgs (np.array): 画像情報

    Returns:
        List: Base64化した画像情報
    """
    img_base64 = []
    for i in imgs[0:5]:
        buff = BytesIO()
        img = Image.fromarray(i)
        img.save(buff, format='JPEG')
        _base64 = base64.urlsafe_b64encode(buff.getvalue()).decode('utf-8')
        img_base64.append(_base64)
    return img_base64


def main():
    args = parse_args()
    
    # tf-servingのURL設定
    URL = "http://localhost:8501/v1/models/{}".format(args.model_name)
    if args.version is not None:
        URL += "/versions/{}".format(args.version)

    # データの読み込みとbase64化
    (test_images, test_labels) = load_data()
    print(type(test_images))
    converted_imgs = convert_img2base64(test_images)

    req = {
        'instances': converted_imgs
    }
    data = json.dumps(req)
    headers = {"content-type": "application/json"}
    json_response = requests.post(URL + ':predict', data=data, headers=headers)
    print(json_response)
    print(json_response.json())


if __name__ == "__main__":
    main()
