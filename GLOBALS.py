import import_lib

seed = 0
import_lib.random.seed(seed)
import_lib.np.random.seed(seed)
batch_size = 32
nb_classes = 2
epochs = 30
img_rows, img_cols = 64, 64
checked = False