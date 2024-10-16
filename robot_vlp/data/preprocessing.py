import random
import pickle
from keras import ops
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from robot_vlp.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR

def build_scaler():
    X_limits = np.array([
        [0,0,-180,0,0],
        [7,7,180,7,7]
    ])
    
    X_scaler = MinMaxScaler()
    X_scaler.fit(X_limits)

    return X_scaler

def apply_scaler(X_data, scaler):
    num_windows, win_len, num_features = X_data.shape
    X_scaled = scaler.transform(X_data.reshape(-1,num_features))
    X_scaled = X_scaled.reshape(num_windows, win_len, num_features)
    return X_scaled


def window_data(X,y, overlap = 0.5 ,window_len = 10):
    overlap_samples = int(overlap * window_len)
    start_index = 0
    end_index = window_len
    X_lst = []
    y_lst = []
    while end_index < len(X):
        X_win = X[start_index:end_index]
        y_win = y[start_index:end_index]
        X_lst.append(X_win)
        y_lst.append(y_win[-1])
        start_index += window_len - overlap_samples
        end_index += window_len - overlap_samples
    return np.array(X_lst), np.array(y_lst)

# X_train, y_train = window_data(X, y, overlap = 0.5, window_len = 10)


def train_test_split_list(lst, train_prop = 0.8, valid_prop = 0.2):
    """
    Takes a list, shuffles it, then breaks in into three parts
    """
    random.seed(42)
    random.shuffle(lst)
    train_prop = 0.6
    valid_prop = 0.2
    test_prop = 1 - (train_prop + valid_prop)
    num_train_elements = int(train_prop * len(lst))
    num_valid_elements = int(test_prop*len(lst))
    num_test_elements = int(valid_prop*len(lst))

    train_elements = [lst.pop() for i in range(num_train_elements)]
    valid_elements = [lst.pop() for i in range(num_valid_elements)]
    test_elements = [lst.pop() for i in range(len(lst))]
    
    return train_elements, valid_elements, test_elements


def load_data(file_list):
    x_lst = []
    y_lst = []
    for file_name in file_list:
        data = pickle.load(open(file_name, 'rb'))
        y_lst.append(data['y'])
        x_lst.append(data['X'])

    return x_lst, y_lst
        

def create_windows(x, y, overlap = 0.5, window_len = 10):
    x_lst = []
    y_lst = []
    # for file_name in file_list:
    #     data = pickle.load(open(file_name, 'rb'))
    #     scaled_X = X_scaler.transform(data['X'])
    for i in range(len(x)):

        x_win, y_win = window_data(x[i], y[i], overlap = overlap ,window_len = window_len)
        # x_win, y_win = window_data(scaled_X, data['y'], overlap = 0.5 ,window_len = 10)
        x_lst.append(x_win)
        y_lst.append(y_win)

    # X_labels = ["encoder_x_hist", "encoder_y_hist", "encoder_heading_hist", "vlp_x_hist", "vlp_y_hist"]
    # y_labels = ["x_hist", "y_hist", "heading_hist"]
    
    X = np.concatenate(x_lst)
    y = np.concatenate(y_lst)

    return X , y

def ang_to_vector(ang, unit = 'radians'):
    if unit == 'degrees':
        ang = ang /180 * np.pi
    x = ops.sin(ang)
    y = ops.cos(ang)
    return ops.stack([x,y], axis = 1)

def vector_to_ang(vector , unit = 'radians'):
    x = vector[:,0]
    y = vector[:,1]
    theta = ops.arctan2(x,y)
    if unit == 'degrees':
         return theta* 180 / np.pi
    elif unit == 'radians':
        return theta
    
def ang_loss_fn(y_true, y_pred):
    return keras.losses.cosine_similarity(y_true, y_pred) + 1


def build_train_data():
    file_list = [file_name for file_name in INTERIM_DATA_DIR.iterdir() if file_name.stem[0] != "."]
    train_files, valid_files, test_files = train_test_split_list(file_list)

    X_scaler = build_scaler()

    X_train_data, y_train_data = load_data(train_files)
    X_valid_data, y_valid_data = load_data(valid_files)
    X_test_data, y_test_data = load_data(test_files)

    X_train_window, y_train = create_windows(X_train_data, y_train_data)
    X_valid_window, y_valid = create_windows(X_valid_data, y_valid_data)
    X_test_window, y_test = create_windows(X_test_data, y_test_data)

    X_train = apply_scaler(X_train_window, X_scaler)
    X_valid = apply_scaler(X_valid_window, X_scaler)
    X_test = apply_scaler(X_test_window, X_scaler)

    data_dic = {
        'train_files':train_files,
        'valid_files':valid_files,
        'test_files':test_files,
        'X_train_data':X_train_data,
        'X_valid_data':X_valid_data,
        'X_test_data':X_test_data,
        'y_train_data':y_train_data,
        'y_valid_data':y_valid_data,
        'y_test_data':y_test_data,
        'X_train':X_train,
        'X_valid':X_valid,
        'X_test':X_test,
        'y_train':y_train,
        'y_valid':y_valid,
        'y_test':y_test
    }

    with open(PROCESSED_DATA_DIR/'model_train_test_data.pickle', 'wb') as handle:
        pickle.dump(data_dic, handle)




if __name__ == "__main__":
    build_train_data()