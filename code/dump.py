

"""
# create our training data from the tweets
x = tr_data_df["Tweet_Text"]
# index all the sentiment labels
y = tr_data_df["Party"]
"""

"""
X = tokenizer.texts_to_sequences(x.values)
X = pad_sequences(X, maxlen=250)
print('Shape of data tensor:', X.shape)

Y = pd.get_dummies(y).values
print('Shape of label tensor:', Y.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.10, random_state = 42)
print("Train (X, Y):", X_train.shape,Y_train.shape)
print("test (X, Y):", X_test.shape,Y_test.shape)
"""


"""
#Keras
model = Sequential()
model.add(Embedding(50000, 100, input_length=X.shape[1]))
model.add(Dropout(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', "binary_accuracy"])

epochs = 1
batch_size = 64

history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

model.save('./LSTM.h5')
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
"""