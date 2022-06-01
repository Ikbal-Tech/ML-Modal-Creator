import json
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
import pandas as pd


def keyValueDecomposition(key):
    index = 0
    if(key == 'N1'):
        index = 0
    elif(key == 'N2'):
        index = 1
    elif(key == 'N3'):
        index = 2
    elif(key == 'N4'):
        index = 3
    elif(key == 'N5'):
        index = 4
    elif(key == 'N6'):
        index = 5
    elif(key == 'N7'):
        index = 6
    elif(key == 'N8'):
        index = 7
    elif(key == 'N9'):
        index = 8
    elif(key == 'N10'):
        index = 9
    elif(key == 'N11'):
        index = 10
    elif(key == 'N12'):
        index = 11
    elif(key == 'N13'):
        index = 12
    elif(key == 'N14'):
        index = 13
    elif(key == 'N15'):
        index = 14
    elif(key == 'COLOR'):
        index = 15
    elif(key == 'id'):
        index = -1
    return index


def readAndParse(jsonFilePath):
    with open(jsonFilePath) as json_file:
        data = json.load(json_file)
        arrayOfArrays = []
        for key, value in data.items():
            # array with 15 elements
            array = [-1, -1, -1, -1, -1, -1, -
                     1, -1, -1, -1, -1, -1, -1, -1, -1]
            indexValue = 0
            firstVal = None
            for key2, value2 in value.items():
                indexValue = keyValueDecomposition(key2)
                if(indexValue == 0):
                    firstVal = value2
                if(indexValue == -1):  # This is the key value of id and it will be ignored
                    array.pop(14)  # Removing dummy element
                    array.insert(0, firstVal)  # Adding first value at the end
                    break
                array.insert(indexValue, value2)
                if(indexValue == 15):
                    array.insert(indexValue, value2)
                    array.pop(indexValue-1)
                    continue
                else:
                    array.pop(indexValue-1)

            arrayOfArrays.append(array)

    print("Done reading json file")
    return arrayOfArrays


def create_train(data, outputFileName):
    x_train = []
    y_train = []
    for item in data:
        x_item = []
        for i in range(16):
            if(i == 15):
                y_train.append(item[i])
            else:
                x_item.append(item[i])
        x_train.append(x_item)

    x_train = np.asarray(x_train)
    y_train = tf.one_hot(y_train, 3)
    y_train = np.asarray(y_train)

    # modeli burada created 15 -> 50 -> 50 -> 3 modelin yapisi bu
    model = tf.keras.Sequential()
    model.add(layers.Dense(50, input_shape=[15]))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(50))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(3, activation="softmax"))

    # modeli compile et. optimizer adam, loss, cok class a gore seçildi.
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics="accuracy"
    )

    # burada da train part
    model.fit(
        x=x_train,
        y=y_train,
        epochs=50  # increase this for better accuracy
    )

    model.save(outputFileName)


def load_model_and_predict(inp, modelFileName):
    inp = np.asarray(inp)
    model = tf.keras.models.load_model(modelFileName)
    predictions = model.predict(inp)
    red_total = 0
    green_total = 0
    blue_total = 0
    for pred in predictions:
        red_total += pred[0]*100
        green_total += pred[1]*100
        blue_total += pred[2]*100
        # print("prediction (r/g/b)%:",pred[0]*100, pred[1]*100, pred[2]*100)

    inp_x_dim, _ = inp.shape
    red_avg = red_total/inp_x_dim
    green_avg = green_total/inp_x_dim
    blue_avg = blue_total/inp_x_dim

    return (red_avg, green_avg, blue_avg)


# Main method
if __name__ == "__main__":
    firebaseFilePath = 'firebase_data.json'
    modalDataPath = 'dummy.modal'
    dummyData = [[3, 2, 5, 4, 6, 2, 34, 5, 2, 3, 5, 7, 2, 3, 5]]

    data = readAndParse(firebaseFilePath)
    create_train(data, modalDataPath)
    prediction = load_model_and_predict(dummyData, modalDataPath)
