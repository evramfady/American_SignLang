import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import numpy as np

# Load data from pickle file
data_dict = pickle.load(open('./data.pickle', 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Set up k-fold cross-validation
k = 5
kf = KFold(n_splits=k, shuffle=True)

# Train and evaluate model on each fold
scores = []
for train_index, test_index in kf.split(data):
    x_train, x_test = data[train_index], data[test_index]
    y_train, y_test = labels[train_index], labels[test_index]

    # Train a random forest classifier on the training data
    model = RandomForestClassifier()
    model.fit(x_train, y_train)

    # Evaluate the model on the testing data
    y_predict = model.predict(x_test)
    score = accuracy_score(y_predict, y_test)
    scores.append(score)

# Compute and print the average accuracy across all folds
avg_score = np.mean(scores)
print('{}% of samples were classified correctly !'.format(avg_score * 100))

# Save the trained model to a pickle file
f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()