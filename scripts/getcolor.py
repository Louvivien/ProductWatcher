from sklearn.cluster import KMeans
import numpy as np
import webcolors
from PIL import Image
import requests
from io import BytesIO
import scipy

# Not working yet
# Add this to requirements.txt:



def closest_color(requested_color):
    min_colors = {}
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_color[0]) ** 2
        gd = (g_c - requested_color[1]) ** 2
        bd = (b_c - requested_color[2]) ** 2
        min_colors[(rd + gd + bd)] = name
    return min_colors[min(min_colors.keys())]

def get_image_color(image_url):
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    img = img.resize((50,50)) # optional, to reduce time
    ar = np.asarray(img)
    shape = ar.shape
    ar = ar.reshape(np.product(shape[:2]), shape[2]).astype(float)

    codes, dist = scipy.cluster.vq.kmeans(ar, 5) # change number to increase color accuracy

    vecs, dist = scipy.cluster.vq.vq(ar, codes)         # assign codes
    counts, bins = np.histogram(vecs, len(codes))    # count occurrences

    index_max = np.argmax(counts)                    # find most frequent
    peak = codes[index_max]
    peak = peak.astype(int)

    colour = closest_color(peak)
    return colour

# Test the function
image_url = 'https://reoriginal.com/image/cache/webp/catalog/product_photo/18087/sumka-herbag-zip-31-2-229x344.webp'
print(get_image_color(image_url))
