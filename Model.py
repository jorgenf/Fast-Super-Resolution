import os
import keras

def normalize_data(data):
    X_train = data["X_train"] / 255
    Y_train = data["Y_train"] / 255
    X_test = data["X_test"] / 255
    Y_test = data["Y_test"] / 255
    return X_train, Y_train, X_test, Y_test

def get_model_dimension(model):
    return model.layers[0].get_input_at(0).get_shape().as_list()[1]

def create_model_directory(directory, model_type, dim, model_alias):
    dir = f"{directory}/saved_models/{model_type}/{dim}_{model_alias}"
    try:
        os.mkdir(dir)
    except:
        print(f"Failed to create directory \"{dir}\"")

    # Creates directory for logs
    log_dir = dir + "/logs/fit/"

    return dir, log_dir

def save_model_info(dir, data, model, epochs, batch_size):
    try:
        with open(dir + "/assets/summary.txt", 'w') as info:
            info.write("Training dataset: " + data["name"] + "\tsize: " + str(len(data["X_train"]) + len(data["X_test"])) + "\n")
            info.write("Epochs: " + str(epochs) + "\tBatch size: " + str(batch_size) + "\n")
            model.summary(print_fn=lambda x: info.write(x + '\n'))
            info.close()
        print(f"Model information saved to {dir}/assets/summary.txt")
    except:
        print(f"Unable to save model information to {dir}/assets/summary.txt")

def load_model(dir):
    return keras.models.load_model(dir)