from sklearn.cluster import KMeans
import numpy as np
import webcolors
from PIL import Image
import requests
from io import BytesIO
import scipy


def closest_color(requested_color):
    colors = {
        "Beige": [223, 196, 163],
        "Black": [0, 0, 0],
        "Blue": [104, 160, 242],
        "Brown": [165, 119, 56],
        "Burgundy": [160, 69, 81],
        "Camel": [198, 151, 90],
        "Charcoal": [78, 78, 78],
        "Ecru": [248, 249, 236],
        "Gold*": [255, 223, 0],  # approximate
        "Green": [137, 179, 121],
        "Grey": [197, 197, 197],
        "Khaki": [151, 164, 99],
        "Metallic*": [192, 192, 192],  # approximate
        # "multicolour": [255, 255, 255],  # white for multicolour
        "Navy": [50, 91, 140],
        "Orange": [255, 180, 73],
        "Pink": [253, 192, 215],
        "Purple": [191, 85, 189],
        "Red": [221, 82, 82],
        "Silver*": [192, 192, 192],  # approximate
        "Turquoise": [141, 228, 221],
        "White": [255, 255, 255],
        "Yellow": [250, 236, 103]
    }
    min_colors = {}
    for key, value in colors.items():
        r_c, g_c, b_c = value
        rd = (r_c - requested_color[0]) ** 2
        gd = (g_c - requested_color[1]) ** 2
        bd = (b_c - requested_color[2]) ** 2
        min_colors[(rd + gd + bd)] = key
    return min_colors[min(min_colors.keys())]


def get_image_color(image_url, border_width=10):
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    img = img.resize((50,50)) # optional, to reduce time

    # Crop the image to remove the border
    width, height = img.size
    img = img.crop((border_width, border_width, width - border_width, height - border_width))

    ar = np.asarray(img)
    shape = ar.shape
    ar = ar.reshape(np.product(shape[:2]), shape[2]).astype(float)

    codes, dist = scipy.cluster.vq.kmeans(ar, 5) # change number to increase color accuracy

    vecs, dist = scipy.cluster.vq.vq(ar, codes)         # assign codes
    counts, bins = scipy.histogram(vecs, len(codes))    # count occurrences

    index_max = scipy.argmax(counts)                    # find most frequent
    peak = codes[index_max]
    peak = peak.astype(int)

    colour = closest_color(peak)
    return colour


# # Test the function
# image_url = 'https://reoriginal.com/image/cache/webp/catalog/product_photo/18087/sumka-herbag-zip-31-2-229x344.webp'
# print(get_image_color(image_url))
