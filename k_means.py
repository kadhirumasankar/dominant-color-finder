from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
from PIL import Image
from matplotlib import style
style.use('ggplot')
import numpy as np
from progress.bar import Bar

im = Image.open('snapseed.png')
(width, height) = im.size
desired_height = 50
resize_factor = desired_height / height
im = im.resize((int(width * resize_factor), int(height * resize_factor)))
(width, height) = im.size

pixel_list = []

bar = Bar('Processing', max=height * width)
for row in range(height):
    for col in range(width):
        (r, g, b) = im.getpixel((col, row))
        pixel_list.append(im.getpixel((col, row)))
        im.putpixel((col, row), (127, 255, 0))
        bar.next()
im.save('checked.png')
bar.finish()

X = np.array(pixel_list)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X[:,0],X[:,1],X[:,2],c=X/255)
ax.set_xlabel('R')
ax.set_ylabel('G')
ax.set_zlabel('B')

plt.show()
