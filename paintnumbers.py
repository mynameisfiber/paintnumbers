import cv2
from matplotlib import pyplot as plt


def nothing(x):
    pass


def plot_image(title, image):
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.xticks([])
    plt.yticks([])


if __name__ == "__main__":
    import sys
    img = cv2.imread(sys.argv[1], 0)
    img = cv2.resize(img, (1024, 768))
    cv2.namedWindow('image')
    cv2.createTrackbar('t1', 'image', 0, 1000, nothing)
    cv2.createTrackbar('t2', 'image', 0, 1000, nothing)
    cv2.createTrackbar('asize', 'image', 0, 2, nothing)
    cv2.createTrackbar('l2grad', 'image', 0, 1, nothing)

    cv2.imshow('image', img)
    while True:
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

        # get current positions of four trackbars
        t1 = cv2.getTrackbarPos('t1', 'image')
        t2 = cv2.getTrackbarPos('t2', 'image')
        asize = 3 + 2 * cv2.getTrackbarPos('asize', 'image')
        l2grad = cv2.getTrackbarPos('l2grad', 'image')
        edges = cv2.Canny(img, t1, t2, apertureSize=asize, L2gradient=l2grad)
        cv2.imshow('image', edges)
