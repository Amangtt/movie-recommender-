import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Layer
from keras.saving import register_keras_serializable
from preprocessor import preprocess

# Define the custom L2Normalize layer
@register_keras_serializable()  # Register the custom layer for serialization
class L2Normalize(Layer):
    def __init__(self, **kwargs):
        super(L2Normalize, self).__init__(**kwargs)

    def call(self, inputs):
        return tf.linalg.l2_normalize(inputs, axis=1)

# Load data
user = pd.read_csv('./Data/content_user_train.csv')
movies = pd.read_csv('./Data/content_item_train.csv')
rating = pd.read_csv('./Data/content_y_train.csv')

def train(user, movies, rating):
    num_user_features = user.shape[1] - 3  # remove userid, rating count, and ave rating during training
    num_item_features = movies.shape[1] - 1  # remove movie id at train time

    u_s = 3  # start of columns to use in training, user
    i_s = 1
    user_train, user_test, movie_train, movie_test, y_train, y_test, scalerUser, scalerItem,scalerTarget = preprocess(user, movies, rating)
    
    tf.random.set_seed(1)
    
    # Define the user neural network
    User_NN = tf.keras.models.Sequential([
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(32)
    ])
    
    # Define the movies neural network
    movies_NN = tf.keras.models.Sequential([
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(32)
    ])
    
    # Define the input layers
    input_user = tf.keras.layers.Input(shape=(num_user_features,))
    input_item = tf.keras.layers.Input(shape=(num_item_features,))
    
    # Pass inputs through the networks
    vu = User_NN(input_user)
    vm = movies_NN(input_item)
    
    # Normalize the outputs using the custom layer
    vu = L2Normalize()(vu)
    vm = L2Normalize()(vm)
    
    # Compute the dot product
    output = tf.keras.layers.Dot(axes=1)([vu, vm])
    
    # Define the model
    model = tf.keras.Model([input_user, input_item], output)
    
    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
                  loss=tf.keras.losses.MeanSquaredError())
    
    # Train the model
    model.fit([user_train[:, u_s:], movie_train[:, i_s:]], y_train, epochs=30)
    
    # Save the model
    model.save("./model/model.keras", save_format="keras")

train(user, movies, rating)