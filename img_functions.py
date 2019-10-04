import import_lib
import GLOBALS

def check_classes(Y):
    if (GLOBALS.checked == False):
        print("-- CLASSES --")
        print(import_lib.Counter(Y).keys())
        print(import_lib.Counter(Y).values())
        GLOBALS.checked = True
        print("-------------")

def rgb_equalization(image):
    channels = import_lib.cv2.split(image)
    eq_channels = []
    for ch, color in zip(channels, ['B', 'G', 'R']):
        eq_channels.append(import_lib.cv2.equalizeHist(ch))

    eq_image = import_lib.cv2.merge(eq_channels)
    eq_image = import_lib.cv2.cvtColor(eq_image, import_lib.cv2.COLOR_BGR2RGB)
    return eq_image

def hsv_equalization(image):
    H, S, V = import_lib.cv2.split(import_lib.cv2.cvtColor(image, import_lib.cv2.COLOR_BGR2HSV))
    eq_V = import_lib.cv2.equalizeHist(V)
    eq_image = import_lib.cv2.cvtColor(import_lib.cv2.merge([H, S, eq_V]), import_lib.cv2.COLOR_HSV2RGB)
    return eq_image
  
def yuv_equalization(image):
    img_yuv = import_lib.cv2.cvtColor(image, import_lib.cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = import_lib.cv2.equalizeHist(img_yuv[:,:,0])
    eq_image = import_lib.cv2.cvtColor(img_yuv, import_lib.cv2.COLOR_YUV2RGB)
    return eq_image

def load_data_eq():
    meta_data = import_lib.pd.read_csv('data/messidor/train/messidor_annotation.csv')
    Y = meta_data['Retinopathy grade'].values
    # Transform into binary classificaiton
    if GLOBALS.nb_classes == 2:
        Y[Y > 0] = 1

    n_samples = Y.shape[0]
    X = import_lib.np.empty((n_samples, GLOBALS.img_rows, GLOBALS.img_cols, 3))
  
    for i in range(n_samples):
        filename = './data/messidor/train/{}.jpg'.format(meta_data['Image name'][i])
        img_cv = import_lib.cv2.resize(import_lib.cv2.imread(filename), (GLOBALS.img_rows, GLOBALS.img_cols))
        x = import_lib.img_to_array(img_cv) / 255.0
        X[i] = x.astype('float32')

    input_shape_l = (GLOBALS.img_rows, GLOBALS.img_cols, 3)
    return X, Y, input_shape_l

def equalize_images(images_in, e_type):
    images_out = import_lib.deepcopy(images_in)
    for i in range(images_out.shape[0]):
        arr_image = images_out[i]*255
        image_to_eq = import_lib.array_to_img(arr_image)
        if (e_type == 0):
            images_out[i] = (import_lib.img_to_array(hsv_equalization(import_lib.np.asarray(image_to_eq))) / 255).astype('float32')
        if (e_type == 1):
            images_out[i] = (import_lib.img_to_array(rgb_equalization(import_lib.np.asarray(image_to_eq))) / 255).astype('float32')
        if (e_type == 2):
            images_out[i] = (import_lib.img_to_array(yuv_equalization(import_lib.np.asarray(image_to_eq))) / 255).astype('float32')

    return images_out
    
def adaptive_equalize_images(images_in):
    images_out = import_lib.deepcopy(images_in)

    for i in range(images_out.shape[0]):
        arr_image = images_out[i]
        import_lib.plt.rcParams['font.size'] = 8
        img_adapteq = import_lib.exposure.equalize_adapthist(arr_image, clip_limit=0.03)
        images_out[i] = img_adapteq

    return images_out
    
def load_data():
    meta_data = import_lib.pd.read_csv('data/messidor/train/messidor_annotation.csv')
    Y = meta_data['Retinopathy grade'].values
    # Transform into binary classificaiton
    if GLOBALS.nb_classes == 2:
        Y[Y > 0] = 1
    check_classes(Y)

    n_samples = Y.shape[0]
    X = import_lib.np.empty((n_samples, GLOBALS.img_rows, GLOBALS.img_cols, 3))

    for i in range(n_samples):
        filename = './data/messidor/train/{}.jpg'.format(meta_data['Image name'][i])
        img = import_lib.load_img(filename, target_size=[GLOBALS.img_rows, GLOBALS.img_cols])
        x = import_lib.img_to_array(img) / 255.0
        X[i] = x.astype('float32')

    input_shape_l = (GLOBALS.img_rows, GLOBALS.img_cols, 3)
    return X, Y, input_shape_l