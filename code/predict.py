import math
import numpy as np
import tifffile as tiff
from argparse import ArgumentParser

from train_unet import get_model, normalize
from sklearn.decomposition import PCA
import pickle


def predict(x, model, patch_sz=160, n_classes=5):
    img_height = x.shape[0]
    img_width = x.shape[1]
    n_channels = x.shape[2]
    # make extended img so that it contains integer number of patches
    npatches_vertical = math.ceil(img_height / patch_sz)
    npatches_horizontal = math.ceil(img_width / patch_sz)
    extended_height = patch_sz * npatches_vertical
    extended_width = patch_sz * npatches_horizontal
    ext_x = np.zeros(shape=(extended_height, extended_width, n_channels), dtype=np.float32)
    # fill extended image with mirrors:
    ext_x[:img_height, :img_width, :] = x
    for i in range(img_height, extended_height):
        ext_x[i, :, :] = ext_x[2 * img_height - i - 1, :, :]
    for j in range(img_width, extended_width):
        ext_x[:, j, :] = ext_x[:, 2 * img_width - j - 1, :]

    # now we assemble all patches in one array
    patches_list = []
    for i in range(0, npatches_vertical):
        for j in range(0, npatches_horizontal):
            x0, x1 = i * patch_sz, (i + 1) * patch_sz
            y0, y1 = j * patch_sz, (j + 1) * patch_sz
            patches_list.append(ext_x[x0:x1, y0:y1, :])
    # model.predict() needs numpy array rather than a list
    patches_array = np.asarray(patches_list)
    # predictions:
    patches_predict = model.predict(patches_array, batch_size=4)
    prediction = np.zeros(shape=(extended_height, extended_width, n_classes), dtype=np.float32)
    for k in range(patches_predict.shape[0]):
        i = k // npatches_horizontal
        j = k % npatches_vertical
        x0, x1 = i * patch_sz, (i + 1) * patch_sz
        y0, y1 = j * patch_sz, (j + 1) * patch_sz
        prediction[x0:x1, y0:y1, :] = patches_predict[k, :, :, :]
    return prediction[:img_height, :img_width, :]


def picture_from_mask(mask, threshold=0.0):
    colors = {
        0: [150, 150, 150],  # Buildings
        1: [223, 194, 125],  # Roads & Tracks
        2: [27, 120, 55],    # Trees
        3: [166, 219, 160],  # Crops
        4: [116, 173, 209]   # Water
    }
    z_order = {
        1: 3,
        2: 4,
        3: 0,
        4: 1,
        5: 2
    }
    pict = 255*np.ones(shape=(3, mask.shape[1], mask.shape[2]), dtype=np.uint8)
    for i in range(1, 6):
        cl = z_order[i]
        for ch in range(3):
            pict[ch,:,:][mask[cl,:,:] > threshold] = colors[cl][ch]
    return pict


def main(patch_size, n_classes, model_path, output_filename, output_mapname, test_id, bands, class_weights):

    # model = get_model(n_channels=4)

    # test_id = 'test'
    # test_id = '24'
    img = normalize(tiff.imread('../data/mband/{}.tif'.format(test_id)).transpose([1, 2, 0]))   # make channels last

    n_channels = len(bands)
    if bands[0] == -1:
        n_channels = 8
    print("\t num_channels", n_channels)

    img = img[:, :, bands]

    model = get_model(n_classes, patch_size, n_channels, class_weights)
    model.load_weights(model_path)
    """
    # Apply PCA to the image
    pca_path = "./pca.pkl"
    # open a file, where you stored the pickled data
    file = open(pca_path, 'rb')
    pca = pickle.load(file)
    file.close()
    reshaped_img_m = img.reshape((img.shape[0]*img.shape[1], img.shape[2]))
    reshaped_img_pca = pca.transform(reshaped_img_m)
    img_pca = reshaped_img_pca.reshape((img.shape[0], img.shape[1],
                                        reshaped_img_pca.shape[1]))
    img = img_pca
    """
    mask = predict(img, model, patch_sz=patch_size, n_classes=n_classes).transpose([2,0,1])  # make channels first
    map = picture_from_mask(mask, 0.5)

    tiff.imsave(output_filename, (255*mask).astype('uint8'))
    tiff.imsave(output_mapname, map)


if __name__ == '__main__':

    parser = ArgumentParser()

    # input weights file path
    parser.add_argument("-i", "--model_path", dest="model_path",
                        help="file path for model", metavar="./PATH", required=True)

    # patch size
    parser.add_argument("-p", "--patch_size", dest="patch_size",
                        help="Patch size", metavar="160", required=True)

    # number of classes
    parser.add_argument("-n", "--num_classes", dest="num_classes",
                        help="Number of classes", metavar="5", required=True)

    # output file name path
    parser.add_argument("-o", "--output_filepath.tif", dest="output_filename",
                        help="file path for model", metavar="./PATH", required=True)

    # output map name path
    parser.add_argument("-m", "--output_mappath.tif", dest="output_mapname",
                        help="file path for model", metavar="./PATH", required=True)

    # test file name
    parser.add_argument("-t", "--test_id", dest="test_id",
                        help="file path for model", metavar="./PATH", required=True)

    # Class weights
    parser.add_argument('-w', '--class_weights', nargs='+', help='Class weights', required=True, dest="class_weights")

    # bands
    parser.add_argument('-bc', '--bands', nargs='+', help='Bands', required=True, dest="bands")

    args = parser.parse_args()

    class_weights = [float(i) for i in args.class_weights]
    bands = [int(i) for i in args.bands]

    for arg in vars(args):
        print(arg, "-->", getattr(args, arg))

    main(patch_size=int(args.patch_size),
         n_classes=int(args.num_classes),
         model_path=args.model_path,
         output_filename=args.output_filename,
         output_mapname=args.output_mapname,
         test_id=args.test_id,
         bands=bands,
         class_weights=class_weights,
         )
