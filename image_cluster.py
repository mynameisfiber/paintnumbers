import cv2
import sys
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from skimage import measure, morphology, filters
import pylab as py
from tqdm import tqdm

py.ion()


def cartoonify(image, n_labels, blur_d, blur_c, blur_s):
    print("bilateral filter")
    image_blur = cv2.bilateralFilter(
        image,
        blur_d, blur_c, blur_s
    )
    pixels = image_blur.reshape((-1, 3))
    print("kmeans")
    kmeans = MiniBatchKMeans(n_labels)
    kmeans.fit(pixels)
    print("creating image")
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_.astype('uint8')
    return labels.reshape(image.shape[:-1]), centers


def _get_gui_params():
    n_labels = cv2.getTrackbarPos('n_labels', 'image')
    blur_d = cv2.getTrackbarPos('blur_d', 'image')
    blur_c = cv2.getTrackbarPos('blur_c', 'image')
    blur_s = cv2.getTrackbarPos('blur_s', 'image')
    return n_labels, blur_d, blur_c, blur_s


def graphical_params(image):
    new_image = np.zeros_like(image)

    def recalc(*args, **kwargs):
        n_labels, blur_d, blur_c, blur_s = _get_gui_params()
        labels, centers = cartoonify(image, n_labels, blur_d, blur_c, blur_s)
        np.copyto(new_image, centers[labels])
    cv2.namedWindow('image')
    cv2.createTrackbar('n_labels', 'image', 2, 25, recalc)
    cv2.createTrackbar('blur_d', 'image', 5, 200, recalc)
    cv2.createTrackbar('blur_c', 'image', 5, 200, recalc)
    cv2.createTrackbar('blur_s', 'image', 5, 200, recalc)
    while True:
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            params = _get_gui_params()
            cv2.destroyAllWindows()
            return params
        cv2.imshow('image', new_image)


def remove_small_regions(labels, min_size):
    blob_labels = measure.label(labels)
    blob_stats = measure.regionprops(blob_labels)
    selem = np.ones((3, 3))
    new_labels = labels.copy()
    img = np.zeros(labels.shape, 'bool')
    for blob in tqdm(blob_stats, desc="removing small regions"):
        if blob.area < min_size:
            img.fill(False)
            xmin, ymin, xmax, ymax = blob.bbox
            img[xmin:xmax, ymin:ymax] = blob.image
            mask = img ^ morphology.binary_dilation(img, selem)
            new_label = labels[mask].max()
            new_labels[img] = new_label
    return new_labels


def _find_center(image):
    selem = np.ones((3, 3))
    eroded = np.zeros((image.shape[0]+2, image.shape[1]+2))
    eroded[1:-1, 1:-1] = image
    while True:
        test_erode = morphology.binary_erosion(eroded, selem=selem)
        if np.all(test_erode == eroded):
            result = np.unravel_index(eroded.argmax(), eroded.shape)
            return (result[0]-1, result[1]-1)
        eroded = test_erode


def plot_coloring_image(labels, centers):
    lines = filters.sobel(labels) != 0
    blob_labels = measure.label(labels)
    blob_stats = measure.regionprops(blob_labels)
    py.imshow(lines)
    for blob in tqdm(blob_stats, desc="creating coloringbook"):
        point = blob.coords[0]
        label = labels[point[0], point[1]]
        xmin, ymin, _, _ = blob.bbox
        center = _find_center(blob.image)
        py.text(center[1] + ymin, center[0] + xmin, str(label),
                horizontalalignment='center',
                verticalalignment='center')


if __name__ == "__main__":
    image = cv2.imread(sys.argv[1])
    image = cv2.resize(image, (1024, 768))

    #params = graphical_params(image)
    #print(params)
    params = (7, 5, 131, 83)

    labels, centers = cartoonify(image, *params)
    new_image = centers[labels]

    print("Finding blobs")
    blob_labels = measure.label(labels)
    blob_stats = measure.regionprops(blob_labels)

    threshold = new_image.shape[0] * new_image.shape[1] * 0.075**2
    new_labels = remove_small_regions(labels, threshold)

    py.figure()
    py.title("cut small regions")
    py.imshow(centers[new_labels])
    py.savefig("small_regions.png", dpi=300)
    py.figure()
    py.title("coloring book")
    plot_coloring_image(new_labels, centers)
    py.savefig("coloring_book.png", dpi=300)
