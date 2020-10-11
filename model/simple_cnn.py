# -*- coding:utf-8 -*-
import os
import numpy as np
import tensorflow as tf
from typing import List
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import LearningRateScheduler
from PIL import Image


class SimpleCNN():
    def __init__(self, tmp_path: str) -> None:
        """コンストラクタ

        Args:
            tmp_path (str): 一時ファイルの格納場所
        """
        self.model = None
        self.tmp_path = tmp_path
        self.class_names = [
            'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
        ]

        if not os.path.exists(self.tmp_path):
            os.makedirs(self.tmp_path)

    def build(self) -> None:
        """モデルをビルドする"""
        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(28, 28), name='image'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax', name='output')
            ])
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

    def train(self, epoch: int, train_X: np.array, train_Y: np.array, valid_X: np.array, valid_Y: np.array) -> tf.python.keras.callbacks.History:
        """学習処理

        Args:
            epoch (int): エポック数
            train_X (np.array): 学習用画像データ
            train_Y (np.array): 学習用ラベルデータ
            valid_X (np.array): 検証用画像データ
            valid_Y (np.array): 検証用ラベルデータ

        Returns:
            tf.python.keras.callbacks.History: 学習履歴
        """
        if self.model is None:
            self.build()
        
        callbacks = self.__get_callbacks(epoch)
        
        history = self.model.fit(
            train_X, train_Y, epochs=epoch,
            validation_data=(valid_X, valid_Y),
            callbacks=callbacks
        )
        print(type(history))
        return history

    def inference(self, X: np.array):
        """推論処理（未実装）

        Args:
            X (np.array): 入力データ
        """
        pass
    
    def export_SavedModel(self, version_number: str, export_dir: str, weight_path: str = None) -> None:
        """SavedModelにエクスポートする

        Args:
            version_number (str): バージョンナンバー
            export_dir (str): 出力ディレクトリ
            weight_path (str, optional): モデルの重みファイルパス. Defaults to None.
        """
        if not os.path.exists(os.path.join(export_dir, str(version_number))):
            os.makedirs(os.path.join(export_dir, str(version_number)))
        
        if weight_path is not None:
            self.model.load_weights(weight_path)

        tf.saved_model.save(self.model, export_dir='{}/{}'.format(export_dir, version_number), signatures=self.serving)

    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string)])
    def serving(self, input_images):
        def _base64_to_array(img):
            img = tf.io.decode_base64(img)
            img = tf.io.decode_image(img)
            img = tf.image.convert_image_dtype(img, tf.float32) # Include normalize https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/image/convert_image_dtype
            img = tf.reshape(img, (28, 28))
            return img

        imgs = tf.map_fn(_base64_to_array, input_images, dtype=tf.float32)
        predictions = self.model(imgs)

        def _convert_to_label(candidates):
            max_prob = tf.math.reduce_max(candidates)
            idx = tf.where(tf.equal(candidates, max_prob))
            label = tf.squeeze(tf.gather(self.class_names, idx))        
            return label

        return tf.map_fn(_convert_to_label, predictions, dtype=tf.string)

    def __get_callbacks(self, epoch: int) -> List:
        """学習時のコールバックを取得する

        Args:
            epoch (int): エポック数

        Returns:
            List: コールバックのリスト
        """
        def scheduler(epoch):
            lr = 1e-3
            if epoch >= epoch // 2:
                lr = 1e-4
            if epoch >= epoch // 4 * 3:
                lr = 1e-5
            return lr
        return [
            EarlyStopping(
                monitor='val_loss', patience=epoch//4 * 3
            ),
            ModelCheckpoint(
                os.path.join(self.tmp_path, "model.h5"),
                monitor='val_loss', save_best_only=True,
                save_weights_only=True, mode='auto',
                verbose=1
            ),
            CSVLogger(
                os.path.join(self.tmp_path, "history.log")
            ),
            LearningRateScheduler(scheduler)
        ]
