import tensorflow as tf
import pandas as pd
from preprocessor import preprocess
from keras.saving import register_keras_serializable
import joblib
# Define the custom L2Normalize layer
@register_keras_serializable()  # Register the custom layer for serialization
class L2Normalize(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(L2Normalize, self).__init__(**kwargs)

    def call(self, inputs):
        return tf.linalg.l2_normalize(inputs, axis=1)

# Load data
user = pd.read_csv('./Data/content_user_train.csv')
movies = pd.read_csv('./Data/content_item_train.csv')
rating = pd.read_csv('./Data/content_y_train.csv')

def metrics(user, movies, rating):
    u_s = 3  # start of columns to use in training, user
    i_s = 1
    user_train, user_test, movie_train, movie_test, y_train, y_test, scalerUser, scalerItem,scalerTarget = preprocess(user, movies, rating)
    
    # Load the model with custom_objects
    model = tf.keras.models.load_model(
        "./model/model.keras",
        custom_objects={"L2Normalize": L2Normalize}  # Register the custom layer
    )
    joblib.dump(scalerUser,'./model/scalerUser.pkl')
    joblib.dump(scalerItem,'./model/scalerItem.pkl')
    joblib.dump(scalerTarget,'./model/scalerTarget.pkl')
    # Evaluate the model
    model.evaluate([user_test[:, u_s:], movie_test[:, i_s:]], y_test)

metrics(user, movies, rating)