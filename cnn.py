# Import TensorFlow dependencies
from gensim.models import Word2Vec
from gensim import utils
import keras
from keras import models, layers
from keras.wrappers.scikit_learn import KerasClassifier
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding, Conv1D, GlobalMaxPooling1D

# Set CNN hyperparameters
max_features = 6000
maxlen = 4000
embedding_dims = 50
hidden_dims = 200
filters = 400
kernel_size = 5
batch_size = 32
epochs = 5

### Featurizes a note for input to the CNN
### Note is a string version of the note, and word_to_index is a dictionary of acceptable vocabulary
def featurize(note, word_to_index):
    words = utils.simple_preprocess(note)
    indices = [word_to_ndex[word] for word in words if word in word_to_embedding_index]
    return indices


# Convolutional neural network architecture
# Argument label_class defines whether the machine learning task is 'binary' or 'multiclass'
# Arguments max_features, maxlen, embedding_dims, hidden_dims, filters and kernel_size are defined above
# Argument n_classes is the nubmber of label classes included in the machine learning task
def create_network(label_class, max_features, maxlen, embedding_dims, hidden_dims, filters, kernel_size, n_classes):

    # Create model
    model = Sequential()

    # we start off with an efficient embedding layer which maps
    # our vocab indices into embedding_dims dimensions
    model.add(Embedding(max_features,
                        embedding_dims,
                        input_length=maxlen))
    model.add(Dropout(0.2))

    # we add a Convolution1D, which will learn filters
    # word group filters of size filter_length:
    model.add(Conv1D(filters,
                     kernel_size,
                     padding='valid',
                     activation='relu',
                     strides=1))

    # we use max pooling:
    model.add(GlobalMaxPooling1D())

    # We add a vanilla hidden layer:
    model.add(Dense(hidden_dims))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))

    # Set architecture according to model type:
    if label_class == 'binary':
        n_dense = 1
        activation_value = 'sigmoid'
        loss_value = 'binary_crossentropy'
    if label_class == 'multiclass':
        n_dense = n_classes
        activation_value = 'softmax'
        loss_value = 'categorical_crossentropy'

    # We project onto a single unit output layer:
    model.add(Dense(n_dense))
    model.add(Activation(activation_value))

    # Compile neural network
    model.compile(loss=loss_value,
                  optimizer='adam',
                  metrics=['accuracy'])

    # Return compiled network
    return model
