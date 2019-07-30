import numpy as np
import matplotlib.pyplot as plt
import cv2


# imReadAndConvert
#       input: 1.filename - string containing the image filename to read.
#              2.representation - representation code, defining if the output should be either a:
#       output:1. grayscale image
#              2. or an RGB image
def imReadAndConvert(filename, representation):
    # Load image
    image = cv2.imread(filename)
    # normolaize
    image = image * (1. / 255)
    if representation == 1:
        # Grayscale image
        image = RGBtoGray(image)
    else:
        # RGB image
        b, g, r = cv2.split(image)
        image = cv2.merge([r, g, b])

    return image


# Calculate the grayscale values so as to have the same luminance as the original color image
def RGBtoGray(image):
    R = np.array(img[:, :, 0])
    G = np.array(img[:, :, 1])
    B = np.array(img[:, :, 2])

    R = (R * .299)
    G = (G * .587)
    B = (B * .114)

    for i in range(3):
        image[:, :, i] = (R + G + B)

    return image


# Depending on representation paramter display the image
def imDisplay(filename, representation):
    image_2 = cv2.imread(filename)
    image_2 = histogramEqualize(image_2)
    plt.imshow(image_2)
    plt.show()


# transforming a RGB values to YIQ values
def transformRGB2YIQ(image):
    # taking sizes of input to make a new image
    height = image.shape[0]
    width = image.shape[1]
    # creating a new matrix
    YIQ = np.zeros((height, width, 3))
    # splitting each dimension to a matrix (opencv functions)
    b, g, r = cv2.split(image)

    imgY = 0.299 * r + 0.587 * g + 0.114 * b
    imgI = 0.596 * r - 0.275 * g - 0.321 * b
    imgQ = 0.212 * r - 0.523 * g + 0.311 * b

    YIQ = cv2.merge([imgY, imgI, imgQ])
    # saving an image as YIQ
    YIQ = YIQ * (1. / 255)

    return (YIQ)


# transforming a YIQ values to RGB values
def transformYIQ2RGB(image):
    # taking sizes of input to make a new image
    height = image.shape[0]
    width = image.shape[1]

    # creating a new matrix
    RGB = np.zeros((height, width, 3))
    q, i, y = cv2.split(image)
    image = cv2.merge([y, i, q])

    imgR = 1 * y + 0.956 * i + 0.619 * q
    imgG = 1 * y - 0.272 * i - 0.647 * q
    imgB = 1 * y - 1.106 * i + 1.703 * q

    RGB = cv2.merge([imgR, imgG, imgB])

    RGB = RGB * (1. / 255)
    print(RGB)
    # saving an image as RGB
    return (RGB)


# Equalize an input image, if input is an RGB image it equalize it with Y channle of YIQ values,
# otherwise input image is grayscale and do it on gray values
def histogramEqualize(imOrig):
    B, G, R = cv2.split(imOrig)

    # Grayscale image
    if (np.array_equal(R, G) and np.array_equal(G, B)):
        imEq, histOrig, histEq = histogram(imOrig)

    else:
        image = cv2.merge([B, G, R])
        YIQimage = transformRGB2YIQ(imOrig)

        YIQimage = YIQimage.astype(np.float32)

        Q, I, Y = cv2.split(YIQimage)
        YIQimage = cv2.merge([Q, I, Y])
        Y = Y * 255
        imEq, histOrig, histEq = histogram(Y)

    imOrig = cv2.merge([R, G, B])
    showOutput(imOrig, imEq, histOrig, histEq)

    return imEq, histOrig, histEq


def histogram(image):
    flat = image.flatten()
    flat = flat.astype(int)

    histOrig = cv2.calcHist([image], [0], None, [256], [0, 256])
    cdf = histOrig.cumsum()

    nj = (cdf - cdf.min()) * 255
    N = cdf.max() - cdf.min()
    cdf = nj / N
    cdf = cdf.astype('uint8')

    # get the value from cumulative sum for every index in flat, and set that as newImage
    imEq = cdf[flat]

    histEq = cv2.calcHist([imEq], [0], None, [256], [0, 256])
    imEq = np.reshape(imEq, img.shape)

    return imEq, histOrig, histEq


def showOutput(imOrig, imEq, histOrig, histEq):
    # set up side-by-side image display
    fig = plt.figure()
    fig.set_figheight(5)
    fig.set_figwidth(10)

    # display original image
    fig.add_subplot(1, 2, 1)
    plt.imshow(imOrig)

    # display equalized image
    fig.add_subplot(1, 2, 2)
    plt.imshow(imEq)
    fig.suptitle('Left is original photo, Right is equalized photo', fontsize=16)
    plt.show(block=True)

    # set up side-by-side image display
    fig = plt.figure()
    fig.set_figheight(5)
    fig.set_figwidth(10)

    # display original histogram image
    fig.add_subplot(1, 2, 1)
    plt.plot(histOrig)

    # display equalized histogram image
    fig.add_subplot(1, 2, 2)
    plt.plot(histEq)
    fig.suptitle('Left is histogram of original photo, Right is histogram of equalized photo', fontsize=12)
    plt.show(block=True)

    # imOrig - input grayscale or RGB image
    # nQuant - number of intensities your output imQuant image should have.
    # nIter  - maximum number of iterations of the optimization procedure.
    def quantizeImage(imOrig, nQuant, nIter):
        # imOrig = cv2.imread('applegray.jpg', cv2.IMREAD_COLOR)
        plt.imshow(cv2.cvtColor(imOrig, cv2.COLOR_BGR2RGB))
        plt.show()

        img_data = imOrig / 255.0
        img_data = img_data.reshape((-1, 3))

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        img_data = img_data.astype(np.float32)
        compactness, labels, centers = cv2.kmeans(img_data, nQuant, None, criteria, nIter, cv2.KMEANS_RANDOM_CENTERS)

        new_colors = centers[labels].reshape((-1, 3))
        img_recolored = new_colors.reshape(imOrig.shape)
        plt.imshow(cv2.cvtColor(img_recolored, cv2.COLOR_BGR2RGB))
        plt.title('16-color image')
        plt.show()

        return labels, centers

    def main():
        im = imDisplay(filename='apple.jpg', representation=2)

    if __name__ == '__main__':
        main(sys.argv[1:])
