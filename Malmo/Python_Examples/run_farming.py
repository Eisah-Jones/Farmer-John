import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import Dense, Conv1D, Dropout, GaussianNoise
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras.metrics import top_k_categorical_accuracy
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix
import pydot


def top_2_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=2)


dataset = pd.read_csv('test_data2.csv').sample(frac=1)

X = dataset.iloc[:,:32].values
Y = to_categorical(dataset.iloc[:, 32].values, num_classes=32)

sc = StandardScaler()
X = sc.fit_transform(X)



X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3)

classifier = Sequential()

classifier.add(Dense(64, activation='relu',
                     kernel_initializer='random_normal'))

classifier.add(Dense(48, activation='relu',
                     kernel_initializer='random_normal'))

classifier.add(Dense(40, activation='relu',
                     kernel_initializer='random_normal'))

classifier.add(Dense(32, activation='relu',
                     kernel_initializer='random_normal'))

classifier.add(Dense(32, activation='sigmoid',
                     kernel_initializer='random_normal'))

classifier.compile(optimizer='adam', loss='categorical_crossentropy',
                   metrics = [top_2_accuracy, 'categorical_accuracy'])

history = classifier.fit(X_train, Y_train, batch_size = 100, epochs = 10)

plt.plot(history.history['top_2_accuracy'], label = 'top 2')
plt.plot(history.history['categorical_accuracy'], label = 'optimal')
plt.plot(history.history['loss'], label = 'loss')
plt.title('Farming Network Metrics')
plt.xlabel('Epoch')
plt.ylabel('Network Metric Value')
plt.legend()
plt.savefig('decisionAcc.png')
plt.close()
eval_model = classifier.evaluate(X_train, Y_train)
print(eval_model)

x_labels = ['loss', 'top 2\naccuracy', 'optimal\naccuracy']
bar_width = 0.25
x_pos = 0.25

plt.bar([0, 0.75, 1.5], eval_model, bar_width, label='training')


eval_model= classifier.evaluate(X_test, Y_test)
print(eval_model)
plt.bar([0.25, 1, 1.75], eval_model, bar_width, label='testing')
plt.title('Farming Network Metric Evaluation')
plt.xticks([0.125, 0.875, 1.625], x_labels)
plt.legend()
plt.savefig('decisionPerformance.png')
plt.close()

Y_pred = classifier.predict(X_test)
Y_pred = (Y_pred > 0.5)

# INSTALL pydot and GraphViz if not already installed
#plot_model(classifier, to_file='model_image.png')

model_json = classifier.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
classifier.save_weights("model.h5")
