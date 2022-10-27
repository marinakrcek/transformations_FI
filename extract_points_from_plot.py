from PIL import Image
from scipy import *
from pylab import *
from matplotlib import colors
import matplotlib.pyplot as plt
import pandas as pd


def convert_color_to_fault(color):
    if color == 'r' or color == 'red':
        return 'Fail'
    if color == 'b' or color == 'blue':
        return 'Mute'
    if color == 'y' or color == 'yellow' or color == 'orange':
        return 'changing'
    if color == 'g' or color == 'green':
        return 'Pass'
    raise ValueError('wrong color as fault class', color)


root = './pics_to_transform/'
# image_file = 'evolvingGA_picek_batina.png'
image_file = 'maldini.png'
im = array(Image.open(root + image_file))

hh = im.shape[0]
ww = im.shape[1]

# imshow(im)
# print(im[100, 0, :])

# Col = array([255, 0, 0])
# bnd = 30
used_cols = ['r', 'b', 'yellow', 'green']
cols_rgb = [[227, 29, 37], [30, 43, 102], [170, 187, 46], [0, 139, 123]]  # with color picker
# cols_rgb = []
# for c in used_cols:
#     if c == 'green':
#         cols_rgb.append([0, 255, 0, 255])
#         continue
#     cols_rgb.append((255*colors.to_rgba_array(c)).tolist()[0])
points = {}
for k in range(0, len(used_cols)):
    points[used_cols[k]] = []
    if im.shape[2] == 4 and len(cols_rgb[k]) < 4:
        cols_rgb[k] = cols_rgb[k] + [255]
    if im.shape[2] == 3 and len(cols_rgb[k]) > 3:
        cols_rgb[k] = cols_rgb[k][:-1]
#     # if c == 'underleg':
#     #     rgb = [203, 229, 203, 255]
#     # else:
#     #     rgb = 255 * colors.to_rgba_array(c)
#     # rgb = rgb[:len(rgb)-2]
#     rgb = cols_rgb[k]
#     for i in range(hh):
#         for j in range(ww):
#             if np.all(im[i, j, :] == rgb):
#                 points[used_cols[k]].append((i, j))
for i in range(hh):
    for j in range(ww):
        for k in range(0, len(used_cols)):
            rgb = cols_rgb[k]
            if sum(abs(im[i, j, :] - rgb)) < 30:
                points[used_cols[k]].append((i, j))

# yax = linspace(1,0,hh)
yax = linspace(0, 1, hh)
xax = linspace(0, 1, ww)

data = []
for c in used_cols:
    data_col = points[c]
    for yi, xi in data_col:
        # print(xax[xi], yax[yi], c)
        data.append([xax[xi], yax[yi], c])
data = np.array(data)
xs = np.array(data[:, 0], dtype=np.float)
# (((OldValue - OldMin) * (NewMax - NewMin)) / (OldMax - OldMin)) + NewMin
xs = (((np.array(data[:, 0], dtype=np.float) - min(xs)) * (1 - 0)) / (max(xs) - min(xs))) + 0
ys = np.array(data[:, 1], dtype=np.float)
ys = (((np.array(data[:, 1], dtype=np.float) - min(ys)) * (1 - 0)) / (max(ys) - min(ys))) + 0

used_cols.reverse()
for c in used_cols:
    percol = np.where(data[:, 2] == c)
    if c == 'underleg':
        c = 'green'
    plt.scatter(xs[percol].tolist(), ys[percol].tolist(), c=c, label=c, s=2)

plt.legend()
plt.gca().invert_yaxis()
plt.savefig(root + image_file + '_reversed.png', format='png', dpi=300, bbox_inches='tight')
# plt.show()
plt.close()

pdf = pd.DataFrame()
# # maldini
xs = np.array(np.round(xs * 480), dtype=int)
ys = np.array(np.round(ys * 480), dtype=int)
xandyvals = np.arange(0, 24 + 0.05, 0.05)
xs = [xandyvals[x] for x in xs]
ys = [xandyvals[y] for y in ys]

# voltage glitching picek and batina
# xvals = np.arange(-5, -0.05 + 0.05, 0.05)
# yvals = np.arange(2, 150 + 2, 2)
# xs = np.array(np.round(xs * (len(xvals)-1)), dtype=int)
# ys = np.array(np.round(ys * (len(yvals)-1)), dtype=int)
# xs = [xvals[x] for x in xs]
# ys = [yvals[y] for y in ys]


pdf['x'] = xs
pdf['y'] = ys
pdf['STATUS'] = np.array([convert_color_to_fault(cl) for cl in data[:, 2]])
print(pdf.size)
pdf.drop_duplicates(subset=['x', 'y'], keep='first', inplace=True)
print(pdf.size)

pdf.to_json(root + image_file + '.json', indent=4, orient="split")
