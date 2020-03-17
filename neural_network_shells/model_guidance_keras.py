## General Guidlines from Keras for different problems:

#1 Multi-class classification

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=20))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
sdg = SDG(lr=0.01, decay=1e-6, momentun=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sdg, metrics=['accuracy'])


#1 Binary classification 

model.add(Dense(64, activation='relu', input_dim=20))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])