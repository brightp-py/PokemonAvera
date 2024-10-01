import numpy as np
import os
import re

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button, Slider

from PIL import Image
from sklearn import cluster

matplotlib.rc('font', weight = 'bold', size = 18)

NATURES = [
    ['Hardy', 'Lonely', 'Adamant', 'Naughty', 'Brave'],
    ['Bold', 'Docile', 'Impish', 'Lax', 'Relaxed'],
    ['Modest', 'Mild', 'Bashful', 'Rash', 'Quiet'],
    ['Calm', 'Gentle', 'Careful', 'Quirky', 'Sassy'],
    ['Timid', 'Hasty', 'Jolly', 'Naive', 'Serious']
]

extract_num = re.compile(r"^[0-9]+")


def make_overlay_colors(_hue, _sat, n_shades, reach_extreme = False):
    for l in range(n_shades):
        _light = 0.1 + 0.8 * l / n_shades

        if n_shades == 1:
            _light = 0.5
        if reach_extreme and l == 0:
            _light = 0
        elif reach_extreme and l == n_shades - 1:
            _light = 1

        _c = (1 - abs(2 * _light - 1)) * _sat
        _x = _c * (1 - abs((_hue / 60) % 2 - 1))
        _m = _light - _c / 2
        _C = int((_c + _m) * 255)
        _X = int((_x + _m) * 255)
        _m = int(_m * 255)
        if _hue < 60:
            yield _C, _X, _m, 255
        elif _hue < 120:
            yield _X, _C, _m, 255
        elif _hue < 180:
            yield _m, _C, _X, 255
        elif _hue < 240:
            yield _m, _X, _C, 255
        elif _hue < 300:
            yield _X, _m, _C, 255
        else:
            yield _C, _m, _X, 255


def replace_color(old_data, new_data, old_color, new_color):
    dcol = np.abs(old_data[:,:,:3] - old_color)
    valid = (dcol.sum(axis = 2) < 3) * (old_data[:,:,3] > 200)
    new_data[valid] = new_color


class ImageLoader:

    def __init__(self, path, files):
        self.path = path
        self.files = files
        self.ind = 0

        self.files = [f for f in self.files if extract_num.match(f)]
        self.files.sort(
            key = lambda x: int(extract_num.match(x).group(0))
        )

        self.fig = plt.figure()
        self.gs = gridspec.GridSpec(1, 2, figure = self.fig)

        self.gs_left = gridspec.GridSpecFromSubplotSpec(
            2,
            2,
            subplot_spec = self.gs[0]
        )

        self.ax1 = self.fig.add_subplot(self.gs_left[0, 0])
        self.ax2 = self.fig.add_subplot(self.gs_left[0, 1])
        self.ax3 = self.fig.add_subplot(self.gs_left[1, 0])
        self.ax4 = self.fig.add_subplot(self.gs_left[1, 1])

        self.im1 = self.ax1.imshow(np.zeros((288, 288, 3)))
        self.im2 = self.ax2.imshow(np.zeros((288, 288, 3)))
        self.sc1 = self.ax3.scatter([], [])
        self.sc2 = self.ax4.scatter([], [])

        self.ax3.set_xlim((-2, 2))
        self.ax3.set_ylim((-2, 2))
        self.ax4.set_xlim((-2, 2))
        self.ax4.set_ylim((-2, 2))

        self.gs_right = gridspec.GridSpecFromSubplotSpec(
            6,
            5,
            subplot_spec = self.gs[1]
        )

        self.nature_buttons = []
        for x in range(5):
            for y in range(5):
                ax = self.fig.add_subplot(self.gs_right[x, y])
                nbutton = Button(ax, NATURES[x][y])
                nbutton.on_clicked(self._save_and_continue(NATURES[x][y]))
                self.nature_buttons.append(nbutton)
        
        ax_st = self.fig.add_subplot(self.gs_right[5, :2])
        self.slider_threshold = Slider(
            ax_st, "Threshold", 0.2, 3.0,
            valinit = 0.7, dragging = False
        )
        self.slider_threshold.on_changed(self.update_threshold)

        ax_sc = self.fig.add_subplot(self.gs_right[5, 2:4])
        self.slider_chrom_rad = Slider(
            ax_sc, "Rad", 0, 2,
            valinit = 1, dragging = False
        )
        self.slider_chrom_rad.on_changed(self.update_chromatic_radius)
        
        ax_p = self.fig.add_subplot(self.gs_right[5, 4])
        self.pbutton = Button(ax_p, "Pass")
        self.pbutton.on_clicked(self.next)

        with open("loader_ind.txt", 'r') as file:
            self.ind = int(file.read())
        
        self.distance_threshold = 0.7
        self.chrom_rad = 1
        self.segments = {}
        self.pkmn_id = ""
        self.foldername = ""

        self.next()


    def update_threshold(self, val):
        self.distance_threshold = val
        self.ind -= 1
        self.next()
    

    def update_chromatic_radius(self, val):
        self.chrom_rad = val
        self.ind -= 1

        a = 1 + self.chrom_rad
        self.ax3.set_xlim((-a, a))
        self.ax3.set_ylim((-a, a))
        self.ax4.set_xlim((-a, a))
        self.ax4.set_ylim((-a, a))

        self.next()
    

    def draw_segment(self, key, data, color):
        if key not in self.segments:
            self.segments[key] = np.zeros_like(data, dtype = np.uint8)
        ncolor = np.concatenate([color, [255]])
        replace_color(data, self.segments[key], color, ncolor)


    def _retrieve_image(self, imname: str):
        """Retrieve pixel and palette data from an image file.
        ARGS:
            imname: str - Complete filepath to a valid image.
        RETURN:
            data: np.ndarray - Pixel data from 0-255, with 4 channels. int32,
                               size = (288, 288, 4).
        """
        im = Image.open(imname).convert('RGBA')
        data = np.array(im.getdata()).reshape((288, 288, 4))
        imcolors = im.getcolors()
        im.close()
        return data, imcolors


    def decide_to_keep(self, imname):
        data, imcolors = self._retrieve_image(imname)

        palette = np.array([c[1][:3] for c in imcolors if c[1][3] > 200])
        lp = len(palette)
        full_palette = palette.reshape((lp, 3)).astype(float)
        palette = full_palette / 255.0

        Cmax = np.max(palette, axis = 1)
        Cmin = np.min(palette, axis = 1)
        Cargmax = np.argmax(palette, axis = 1)
        d = Cmax - Cmin

        hues = np.zeros_like(d)
        is_R = (d > 0) * (Cargmax == 0)
        is_G = (d > 0) * (Cargmax == 1)
        is_B = (d > 0) * (Cargmax == 2)
        hues[is_R] = np.fmod((palette[is_R, 1] - palette[is_R, 2]) / d[is_R], 6)
        hues[is_G] = (palette[is_G, 2] - palette[is_G, 0]) / d[is_G] + 2
        hues[is_B] = (palette[is_B, 0] - palette[is_B, 1]) / d[is_B] + 4

        sat = np.zeros_like(hues)
        sat[Cmax > 0] = d[Cmax > 0] / Cmax[Cmax > 0]
        angle = hues * 3.14159 / 3
        x = np.cos(angle) * (Cmax + self.chrom_rad - sat / 4)
        y = np.sin(angle) * (Cmax + self.chrom_rad - sat / 4)
        x[d == 0] = 0

        self.sc1.set_offsets(np.c_[x, y])
        self.sc1.set_color(palette)

        c = cluster.AgglomerativeClustering(
            n_clusters = None,
            distance_threshold = self.distance_threshold,
            linkage = 'average'
        )
        pos_data = np.concatenate([x[:,np.newaxis], y[:,np.newaxis]], axis = 1)
        labels = c.fit_predict(pos_data)

        boring_colors = [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0.7, 0.7, 0],
            [0.7, 0, 0.7],
            [0, 0.7, 0.7],
            [1, 0.5, 0],
            [0.5, 1, 0],
            [1, 0, 0.5],
            [0.5, 0, 1],
            [0, 1, 0.5],
            [0, 0.5, 1]
        ]
        self.sc2.set_offsets(np.c_[x, y])
        self.sc2.set_color([boring_colors[l] for l in labels])

        groups = [[] for _ in range(max(labels) + 1)]
        for lab, col in zip(labels, full_palette):
            groups[lab].append(col.astype(int))
        groups.sort(key = len)

        singles = []
        for i, group in enumerate(groups):
            if len(group) > 1:
                break
            singles.extend(groups[0])
        groups = [singles] + groups[i:]
        groups.sort(key = len)

        overlay_keys = [
            (0, 0, 0),
            (45 , 0.5, 16),
            (180, 0.6, 9),
            (315, 0.7, 7),
            (90 , 0.8, 6),
            (225, 0.8, 6),
            (135, 0.9, 3),
            (270, 0.9, 3),
            (90 , 1.0, 3)
        ]

        self.segments = {}
        ndata = data.copy()
        for group in groups:
            group.sort(key = np.sum)
            if any(col.max() == col.min() for col in group):
                for n, o in zip(make_overlay_colors(0, 0, len(group), True), group):
                    replace_color(data, ndata, np.array(o), n)
                    self.draw_segment(0, data, np.array(o))
                continue
            while len(overlay_keys) > 1 and overlay_keys[-1][2] < len(group):
                del overlay_keys[-1]
                # if len(overlay_keys) <= 1:
                #     raise IndexError
            if len(overlay_keys) == 1:
                print("Too many colors!")
            for n, o in zip(make_overlay_colors(*overlay_keys[-1]), group):
                replace_color(data, ndata, np.array(o), n)
                self.draw_segment(len(overlay_keys) - 1, data, np.array(o))
            del overlay_keys[-1]

        self.im1.set_data(data)
        self.im2.set_data(ndata)
        plt.draw()
    

    def save_segments(self, nature: str):
        if self.pkmn_id == "":
            return

        d = f"img/{self.pkmn_id}"
        if not os.path.exists(d):
            os.mkdir(d)
        
        d = f"{d}/{self.foldername}"
        if not os.path.exists(d):
            os.mkdir(d)
        
        for path, _, files in os.walk(d):
            for file in files:
                os.remove(os.path.join(path, file))
        
        for key, data in self.segments.items():
            im = Image.fromarray(data)
            im.save(f"{d}/{key}.png")
        
        with open(f"{d}/data.txt", 'w') as file:
            file.write(nature)
    

    def next(self, _ = None):
        self.filename = self.files[self.ind]
        self.pkmn_id = extract_num.match(self.filename).group(0)
        self.foldername = self.filename.split('.')[0]

        p = os.path.join(self.path, self.filename)
        self.decide_to_keep(p)
        self.ind += 1
        with open("loader_ind.txt", 'w') as file:
            file.write(str(self.ind - 1))
    

    def _save_and_continue(self, nature: str):
        """Create a function to save a nature, to be used on a button click.
        ARGS:
            nature: str - Single string, like "Modest" or "Serious".
        RETURN:
            f: function - Argument to pass to <Button>.on_clicked.
        """
        def f(_):
            self.save_segments(nature)
            self.next(_)
        return f
    

if __name__ == "__main__":
    with open("foldername.txt", 'r') as file:
        folderpath = file.read()
    for path, _, files in os.walk(folderpath):
        loader = ImageLoader(path, files)
        plt.show()
