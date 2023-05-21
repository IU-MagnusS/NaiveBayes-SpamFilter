Problem 3: How to use Predict() method:

# Save the code in a file called naive_bayes_filter.py.
    In a terminal, run the following command:

    python naive_bayes_filter.py
    -> This will create an object called naive_bayes_filter.

# fit the filter with train data
    naive_bayes_filter.fit(X_train, y_train)

# Predict the labels of the test data
    predictions = naive_bayes_filter.predict(X_test[500:505])

# Print the predictions
    print(predictions)
    -> Output: ['ham', 'ham', 'ham', 'ham', 'spam']
    
Problem4: How to Use the Score() method:

# Create the filter
    NB = NaiveBayesFilter()

# Fit the filter with train data
    NB.fit(X_train, y_train)

# Test the predict function with five data points in test data
    predict_labels = NB.predict(X_test[500:505])

# Calculate the score
    recall = NB.score(y_test[500:505], predict_labels)

    print("recall of NB: ", recall)
    -> Output recall of NB: 0.8