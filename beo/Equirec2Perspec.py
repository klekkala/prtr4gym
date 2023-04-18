import os,io
import sys
import cv2
import h5py
import numpy as np
import time
# import leveldb
import plyvel
import cupy as cp
from memory_profiler import profile
from PIL import Image



def str2byte(s):
    s = s.encode()
    return s


def byte2str(b):
    b = b.decode()
    return b


class Equirectangular:
    def __init__(self):
        self.db = ''
        count = 0
        while self.db == '':
            try:
                if os.path.exists('data' + str(count) + '/'):
                    self.db = plyvel.DB('data' + str(count) + '/')
                else:
                    shutil.copytree('data/', 'data' + str(count) + '/')
                    self.db = plyvel.DB('data' + str(count) + '/')
            except IOError:
                count += 1
        self._img=None
        self._height=0
        self._width=0

    def get_image(self, pos):
        pos = str(pos[0]) + ',' + str(pos[1])
        self._img = cv2.imdecode(np.frombuffer(self.db.get(str2byte(pos)), np.uint8), -1)
        [self._height, self._width, _] = self._img.shape



    # @profile(precision=5)
    def GetPerspective(self, FOV, THETA, PHI, height, width, RADIUS=128):
        #
        # THETA is left/right angle, PHI is up/down angle, both in degree
        #

        equ_h = self._height
        equ_w = self._width
        equ_cx = (equ_w - 1) / 2.0
        equ_cy = (equ_h - 1) / 2.0

        wFOV = FOV
        hFOV = float(height) / width * wFOV

        c_x = (width - 1) / 2.0
        c_y = (height - 1) / 2.0

        wangle = (180 - wFOV) / 2.0
        w_len = 2 * RADIUS * cp.sin(cp.radians(wFOV / 2.0)) / cp.sin(cp.radians(wangle))
        w_interval = w_len / (width - 1)

        hangle = (180 - hFOV) / 2.0
        h_len = 2 * RADIUS * cp.sin(cp.radians(hFOV / 2.0)) / cp.sin(cp.radians(hangle))
        h_interval = h_len / (height - 1)
        x_map = cp.zeros([height, width], cp.float32) + RADIUS
        y_map = cp.tile((cp.arange(0, width) - c_x) * w_interval, [height, 1])
        z_map = -cp.tile((cp.arange(0, height) - c_y) * h_interval, [width, 1]).T
        D = cp.sqrt(x_map ** 2 + y_map ** 2 + z_map ** 2)
        xyz = cp.zeros([height, width, 3], float)
        xyz[:, :, 0] = (RADIUS / D * x_map)[:, :]
        xyz[:, :, 1] = (RADIUS / D * y_map)[:, :]
        xyz[:, :, 2] = (RADIUS / D * z_map)[:, :]

        y_axis = cp.array([0.0, 1.0, 0.0], cp.float32)
        z_axis = cp.array([0.0, 0.0, 1.0], cp.float32)
        [R1, _] = cv2.Rodrigues(cp.asnumpy(z_axis * cp.radians(THETA)))
        R1 = cp.array(R1)
        [R2, _] = cv2.Rodrigues(cp.asnumpy(cp.dot(R1, y_axis) * cp.radians(-PHI)))
        R2 = cp.array(R2)
        # R1 = cp.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]])
        # R2 = cp.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]])

        xyz = xyz.reshape([height * width, 3]).T
        xyz = cp.dot(R1, xyz)
        xyz = cp.dot(R2, xyz).T
        lat = cp.arcsin(xyz[:, 2] / RADIUS)
        lon = cp.zeros([height * width], float)
        theta = cp.arctan(xyz[:, 1] / xyz[:, 0])
        idx1 = xyz[:, 0] > 0
        idx2 = xyz[:, 1] > 0

        idx3 = ((1 - idx1) * idx2).astype(bool)
        idx4 = ((1 - idx1) * (1 - idx2)).astype(bool)

        lon[idx1] = theta[idx1]
        lon[idx3] = theta[idx3] + cp.pi
        lon[idx4] = theta[idx4] - cp.pi
        lon = lon.reshape([height, width]) / cp.pi * 180
        lat = -lat.reshape([height, width]) / cp.pi * 180
        lon = lon / 180 * equ_cx + equ_cx
        lat = lat / 90 * equ_cy + equ_cy
        lon = cp.asnumpy(lon)
        lat = cp.asnumpy(lat)


        persp = cv2.remap(self._img, lon.astype(np.float32), lat.astype(np.float32), cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)

        return persp



