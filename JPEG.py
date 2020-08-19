import cv2,math
import numpy as np

class JPEG:

    WR = 0.299
    WG = 0.587
    WB = 0.114
    table0 = \
        [
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99]
        ]
    table1 = \
        [
            [17, 18, 24, 47, 99, 99, 99, 99],
            [18, 21, 26, 66, 99, 99, 99, 99],
            [24, 26, 56, 99, 99, 99, 99, 99],
            [47, 66, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99]
        ]
    DC_Y = {
        0: '00',
        1: '010',
        2: '011',
        3: '100',
        4: '101',
        5: '110',
        6: '1110',
        7: '11110',
        8: '111110',
        9: '1111110',
        10: '11111110',
        11: '111111110'
    }
    DC_UV = {
        0: '00',
        1: '01',
        2: '10',
        3: '110',
        4: '1110',
        5: '11110',
        6: '111110',
        7: '1111110',
        8: '11111110',
        9: '111111110',
        10: '1111111110',
        11: '11111111110'
    }
    AC_Y = {
        (0, 0): '1010',
        (0, 1): '00',
        (0, 2): '01',
        (0, 3): '100',
        (0, 4): '1011',
        (0, 5): '11010',
        (0, 6): '1111000',
        (0, 7): '11111000',
        (0, 8): '1111110110',
        (0, 9): '1111111110000010',
        (0, 10): '1111111110000011',
        (1, 1): '1100',
        (1, 2): '11011',
        (1, 3): '1111001',
        (1, 4): '111110110',
        (1, 5): '11111110110',
        (1, 6): '1111111110000100',
        (1, 7): '1111111110000101',
        (1, 8): '1111111110000110',
        (1, 9): '1111111110000111',
        (1, 10): '1111111110001000',
        (2, 1): '11100',
        (2, 2): '11111001',
        (2, 3): '1111110111',
        (2, 4): '111111110100',
        (2, 5): '1111111110001001',
        (2, 6): '1111111110001010',
        (2, 7): '1111111110001011',
        (2, 8): '1111111110001100',
        (2, 9): '1111111110001101',
        (2, 10): '1111111110001110',
        (3, 1): '111010',
        (3, 2): '111110111',
        (3, 3): '111111110101',
        (3, 4): '1111111110001111',
        (3, 5): '1111111110010000',
        (3, 6): '1111111110010001',
        (3, 7): '1111111110010010',
        (3, 8): '1111111110010011',
        (3, 9): '1111111110010100',
        (3, 10): '1111111110010101',
        (4, 1): '111011',
        (4, 2): '1111111000',
        (4, 3): '1111111110010110',
        (4, 4): '1111111110010111',
        (4, 5): '1111111110011000',
        (4, 6): '1111111110011001',
        (4, 7): '1111111110011010',
        (4, 8): '1111111110011011',
        (4, 9): '1111111110011100',
        (4, 10): '1111111110011101',
        (5, 1): '1111010',
        (5, 2): '11111110111',
        (5, 3): '1111111110011110',
        (5, 4): '1111111110011111',
        (5, 5): '1111111110100000',
        (5, 6): '1111111110100001',
        (5, 7): '1111111110100010',
        (5, 8): '1111111110100011',
        (5, 9): '1111111110100100',
        (5, 10): '1111111110100101',
        (6, 1): '1111011',
        (6, 2): '111111110110',
        (6, 3): '1111111110100110',
        (6, 4): '1111111110100111',
        (6, 5): '1111111110101000',
        (6, 6): '1111111110101001',
        (6, 7): '1111111110101010',
        (6, 8): '1111111110101011',
        (6, 9): '1111111110101100',
        (6, 10): '1111111110101101',
        (7, 1): '11111010',
        (7, 2): '111111110111',
        (7, 3): '1111111110101110',
        (7, 4): '1111111110101111',
        (7, 5): '1111111110110000',
        (7, 6): '1111111110110001',
        (7, 7): '1111111110110010',
        (7, 8): '1111111110110011',
        (7, 9): '1111111110110100',
        (7, 10): '1111111110110101',
        (8, 1): '111111000',
        (8, 2): '111111111000000',
        (8, 3): '1111111110110110',
        (8, 4): '1111111110110111',
        (8, 5): '1111111110111000',
        (8, 6): '1111111110111001',
        (8, 7): '1111111110111010',
        (8, 8): '1111111110111011',
        (8, 9): '1111111110111100',
        (8, 10): '1111111110111101',
        (9, 1): '111111001',
        (9, 2): '1111111110111110',
        (9, 3): '1111111110111111',
        (9, 4): '1111111111000000',
        (9, 5): '1111111111000001',
        (9, 6): '1111111111000010',
        (9, 7): '1111111111000011',
        (9, 8): '1111111111000100',
        (9, 9): '1111111111000101',
        (9, 10): '1111111111000110',
        (10, 1): '111111010',
        (10, 2): '1111111111000111',
        (10, 3): '1111111111001000',
        (10, 4): '1111111111001001',
        (10, 5): '1111111111001010',
        (10, 6): '1111111111001011',
        (10, 7): '1111111111001100',
        (10, 8): '1111111111001101',
        (10, 9): '1111111111001110',
        (10, 10): '1111111111001111',
        (11, 1): '1111111001',
        (11, 2): '1111111111010000',
        (11, 3): '1111111111010001',
        (11, 4): '1111111111010010',
        (11, 5): '1111111111010011',
        (11, 6): '1111111111010100',
        (11, 7): '1111111111010101',
        (11, 8): '1111111111010110',
        (11, 9): '1111111111010111',
        (11, 10): '1111111111011000',
        (12, 1): '1111111010',
        (12, 2): '1111111111011001',
        (12, 3): '1111111111011010',
        (12, 4): '1111111111011011',
        (12, 5): '1111111111011100',
        (12, 6): '1111111111011101',
        (12, 7): '1111111111011110',
        (12, 8): '1111111111011111',
        (12, 9): '1111111111100000',
        (12, 10): '1111111111100001',
        (13, 1): '11111111000',
        (13, 2): '1111111111100010',
        (13, 3): '1111111111100011',
        (13, 4): '1111111111100100',
        (13, 5): '1111111111100101',
        (13, 6): '1111111111100110',
        (13, 7): '1111111111100111',
        (13, 8): '1111111111101000',
        (13, 9): '1111111111101001',
        (13, 10): '1111111111101010',
        (14, 1): '1111111111101011',
        (14, 2): '1111111111101100',
        (14, 3): '1111111111101101',
        (14, 4): '1111111111101110',
        (14, 5): '1111111111101111',
        (14, 6): '1111111111110000',
        (14, 7): '1111111111110001',
        (14, 8): '1111111111110010',
        (14, 9): '1111111111110011',
        (14, 10): '1111111111110100',
        (15, 0): '11111111001',
        (15, 1): '1111111111110101',
        (15, 2): '1111111111110110',
        (15, 3): '1111111111110111',
        (15, 4): '1111111111111000',
        (15, 5): '1111111111111001',
        (15, 6): '1111111111111010',
        (15, 7): '1111111111111011',
        (15, 8): '1111111111111100',
        (15, 9): '1111111111111101',
        (15, 10): '1111111111111110',
    }
    AC_UV = {
        (0, 0): '00',
        (0, 1): '01',
        (0, 2): '100',
        (0, 3): '1010',
        (0, 4): '11000',
        (0, 5): '11001',
        (0, 6): '111000',
        (0, 7): '1111000',
        (0, 8): '111110100',
        (0, 9): '1111110110',
        (0, 10): '111111110100',
        (1, 1): '1011',
        (1, 2): '111001',
        (1, 3): '11110110',
        (1, 4): '111110101',
        (1, 5): '11111110110',
        (1, 6): '111111110101',
        (1, 7): '1111111110001000',
        (1, 8): '1111111110001001',
        (1, 9): '1111111110001010',
        (1, 10): '1111111110001011',
        (2, 1): '11010',
        (2, 2): '11110111',
        (2, 3): '1111110111',
        (2, 4): '111111110110',
        (2, 5): '111111111000010',
        (2, 6): '1111111110001100',
        (2, 7): '1111111110001101',
        (2, 8): '1111111110001110',
        (2, 9): '1111111110001111',
        (2, 10): '1111111110010000',
        (3, 1): '11011',
        (3, 2): '11111000',
        (3, 3): '1111111000',
        (3, 4): '111111110111',
        (3, 5): '1111111110010001',
        (3, 6): '1111111110010010',
        (3, 7): '1111111110010011',
        (3, 8): '1111111110010100',
        (3, 9): '1111111110010101',
        (3, 10): '1111111110010110',
        (4, 1): '111010',
        (4, 2): '111110110',
        (4, 3): '1111111110010111',
        (4, 4): '1111111110011000',
        (4, 5): '1111111110011001',
        (4, 6): '1111111110011010',
        (4, 7): '1111111110011011',
        (4, 8): '1111111110011100',
        (4, 9): '1111111110011101',
        (4, 10): '1111111110011110',
        (5, 1): '111011',
        (5, 2): '1111111001',
        (5, 3): '1111111110011111',
        (5, 4): '1111111110100000',
        (5, 5): '1111111110100001',
        (5, 6): '1111111110100010',
        (5, 7): '1111111110100011',
        (5, 8): '1111111110100100',
        (5, 9): '1111111110100101',
        (5, 10): '1111111110100110',
        (6, 1): '1111001',
        (6, 2): '11111110111',
        (6, 3): '1111111110100111',
        (6, 4): '1111111110101000',
        (6, 5): '1111111110101001',
        (6, 6): '1111111110101010',
        (6, 7): '1111111110101011',
        (6, 8): '1111111110101100',
        (6, 9): '1111111110101101',
        (6, 10): '1111111110101110',
        (7, 1): '1111010',
        (7, 2): '11111111000',
        (7, 3): '1111111110101111',
        (7, 4): '1111111110110000',
        (7, 5): '1111111110110001',
        (7, 6): '1111111110110010',
        (7, 7): '1111111110110011',
        (7, 8): '1111111110110100',
        (7, 9): '1111111110110101',
        (7, 10): '1111111110110110',
        (8, 1): '11111001',
        (8, 2): '1111111110110111',
        (8, 3): '1111111110111000',
        (8, 4): '1111111110111001',
        (8, 5): '1111111110111010',
        (8, 6): '1111111110111011',
        (8, 7): '1111111110111100',
        (8, 8): '1111111110111101',
        (8, 9): '1111111110111110',
        (8, 10): '1111111110111111',
        (9, 1): '111110111',
        (9, 2): '1111111111000000',
        (9, 3): '1111111111000001',
        (9, 4): '1111111111000010',
        (9, 5): '1111111111000011',
        (9, 6): '1111111111000100',
        (9, 7): '1111111111000101',
        (9, 8): '1111111111000110',
        (9, 9): '1111111111000111',
        (9, 10): '1111111111001000',
        (10, 1): '111111000',
        (10, 2): '1111111111001001',
        (10, 3): '1111111111001010',
        (10, 4): '1111111111001011',
        (10, 5): '1111111111001100',
        (10, 6): '1111111111001101',
        (10, 7): '1111111111001110',
        (10, 8): '1111111111001111',
        (10, 9): '1111111111010000',
        (10, 10): '1111111111010001',
        (11, 1): '111111001',
        (11, 2): '1111111111010010',
        (11, 3): '1111111111010011',
        (11, 4): '1111111111010100',
        (11, 5): '1111111111010101',
        (11, 6): '1111111111010110',
        (11, 7): '1111111111010111',
        (11, 8): '1111111111011000',
        (11, 9): '1111111111011001',
        (11, 10): '1111111111011010',
        (12, 1): '111111010',
        (12, 2): '1111111111011011',
        (12, 3): '1111111111011100',
        (12, 4): '1111111111011101',
        (12, 5): '1111111111011110',
        (12, 6): '1111111111011111',
        (12, 7): '1111111111100000',
        (12, 8): '1111111111100001',
        (12, 9): '1111111111100010',
        (12, 10): '1111111111100011',
        (13, 1): '11111111001',
        (13, 2): '1111111111100100',
        (13, 3): '1111111111100101',
        (13, 4): '1111111111100110',
        (13, 5): '1111111111100111',
        (13, 6): '1111111111101000',
        (13, 7): '1111111111101001',
        (13, 8): '1111111111101010',
        (13, 9): '1111111111101011',
        (13, 10): '1111111111101100',
        (14, 1): '11111111100000',
        (14, 2): '1111111111101101',
        (14, 3): '1111111111101110',
        (14, 4): '1111111111101111',
        (14, 5): '1111111111110000',
        (14, 6): '1111111111110001',
        (14, 7): '1111111111110010',
        (14, 8): '1111111111110011',
        (14, 9): '1111111111110100',
        (14, 10): '1111111111110101',
        (15, 0): '1111111010',
        (15, 1): '111111111000011',
        (15, 2): '1111111111110110',
        (15, 3): '1111111111110111',
        (15, 4): '1111111111111000',
        (15, 5): '1111111111111001',
        (15, 6): '1111111111111010',
        (15, 7): '1111111111111011',
        (15, 8): '1111111111111100',
        (15, 9): '1111111111111101',
        (15, 10): '1111111111111110',
    }
    DC_Y2 = {value: key for key, value in DC_Y.items()}
    DC_UV2 = {value: key for key, value in DC_UV.items()}
    AC_Y2 = {value: key for key, value in AC_Y.items()}
    AC_UV2 = {value: key for key, value in AC_UV.items()}

    def printBlock(self, block):
        for row in block:
            print(row)
    def rgb2yuv(self, img, width, height):
        Y = [[0 for i in range(width)] for i in range(height)]
        U = [[0 for i in range((width - 1) // 2 + 1)] for i in range((height - 1) // 2 + 1)]
        V = [[0 for i in range((width - 1) // 2 + 1)] for i in range((height - 1) // 2 + 1)]
        for i in range(height):
            flag = False
            if i % 2 == 1 or i == height - 1:
                flag = True
            for j in range(width):
                B = img[i][j][0]
                G = img[i][j][1]
                R = img[i][j][2]
                Y[i][j] = (self.WR * R + self.WG * G + self.WB * B)
                if i % 2 == 0 and j % 2 == 0:
                    U[i // 2][j // 2] = 0.5 *(((-self.WR)/(1-self.WB)) * R + ((-self.WG)/(1-self.WB)) * G + B)
                if flag:
                    V[i // 2][j // 2] = 0.5 *(((-self.WB)/(1-self.WR)) * B + ((-self.WG)/(1-self.WR)) * G + R)
        return Y, U, V 
    def yuv2rgb(self, Y, U, V, width, height):  
        img = np.zeros([height, width, 3], dtype=np.uint8)
        for i in range(height):
            for j in range(width):
                img[i][j][0] = np.clip(Y[i][j]+2*(1-self.WB) * (U[i//2][j//2]),0,255 )#  B
                img[i][j][1] = np.clip(Y[i][j]-(2*(1-self.WB)*self.WB/self.WG)*(U[i//2][j//2])-(2*(1-self.WR)*self.WR/self.WG)*(V[i//2][j//2] ),0,255)
                img[i][j][2] = np.clip(Y[i][j]+2*(1-self.WR) * (V[i//2][j//2]),0,255 )
        return img

    def fill(self, img):
        width = len(img[0])
        height = len(img)
        if height % 8 != 0:
            for i in range(8 - height % 8):
                img.append([0 for i in range(width)])
        if width % 8 != 0:
            for row in img:
                for i in range(8 - width % 8):
                    row.append(0)
        return img
    def split(self, img):
        width = len(img[0])
        height = len(img)
        blocks = []
        for i in range(height // 8):
            for j in range(width // 8):
                temp = [[0 for i in range(8)] for i in range(8)]
                for r in range(8):
                    for c in range(8):
                        temp[r][c] = img[i * 8 + r][j * 8 + c]
                blocks.append(temp)
        return blocks
    def merge(self, imgY, imgU, imgV, height, width):
        Yheight = (height - 1) // 8 + 1
        Ywidth = (width - 1) // 8 + 1
        UVheight = (height - 1) // 2 // 8 + 1
        UVwidth = (width - 1) // 2 // 8 + 1
        Y = [[0 for i in range(Ywidth * 8)] for i in range(Yheight * 8)]
        U = [[0 for i in range(UVwidth * 8)] for i in range(UVheight * 8)]
        V = [[0 for i in range(UVwidth * 8)] for i in range(UVheight * 8)]
        for i in range(Yheight):
            for j in range(Ywidth):
                for r in range(8):
                    for c in range(8):
                        Y[i * 8 + r][j * 8 + c] = imgY[i * Ywidth + j][r][c]
        for i in range(UVheight):
            for j in range(UVwidth):
                for r in range(8):
                    for c in range(8):
                        U[i * 8 + r][j * 8 + c] = imgU[i * UVwidth + j][r][c]
                        V[i * 8 + r][j * 8 + c] = imgV[i * UVwidth + j][r][c]
        while len(Y) > height:
            Y.pop()
        for row in Y:
            while len(row) > width:
                row.pop()
        while len(U) > (height - 1) // 2 + 1:
            U.pop()
            V.pop()
        for row in U:
            while len(row) > (width - 1) // 2 + 1:
                row.pop()
        for row in V:
            while len(row) > (width - 1) // 2 + 1:
                row.pop()

        return Y, U, V
    def FDCT(self, block):
        temp = [[0 for i in range(8)] for i in range (8)]
        for u in range(8):
            for v in range(8):
                n = 0
                for i in range(8):
                    for j in range(8):
                        n += math.cos((2 * i + 1) * u * math.pi / 16) * math.cos((2 * j + 1) * v * math.pi / 16) * \
                             block[i][j]
                temp[u][v] = round(self.C(u) * self.C(v) / 4 * n)
        return temp
    def IDCT(self, block):
        temp = [[0 for i in range(8)] for i in range(8)]
        for i in range(8):
            for j in range(8):
                n = 0
                for u in range(8):
                    for v in range(8):
                        n += math.cos((2 * i + 1) * u * math.pi / 16) * math.cos((2 * j + 1) * v * math.pi / 16) * \
                             block[u][v] * self.C(u) * self.C(v) / 4
                temp[i][j] = round(n)
        return temp
    def C(self, n):
        if n == 0:
            return pow(2, 1/2)/2
        else:
            return 1

    def quanY(self, img):
        temp = [[0 for i in range(8)] for i in range(8)]
        for i in range(8):
            for j in range(8):
                temp[i][j] = round(img[i][j] / self.table0[i][j])
        return temp
    def quanUV(self, img):
        temp = [[0 for i in range(8)] for i in range(8)]
        for i in range(8):
            for j in range(8):
                temp[i][j] = round(img[i][j] / self.table1[i][j])
        return temp
    def reY(self, img):
        temp = [[0 for i in range(8)] for i in range(8)]
        for i in range(8):
            for j in range(8):
                temp[i][j] = round(img[i][j] * self.table0[i][j])
        return temp
    def reUV(self, img):
        temp = [[0 for i in range(8)] for i in range(8)]
        for i in range(8):
            for j in range(8):
                temp[i][j] = round(img[i][j] * self.table1[i][j])
        return temp

    def ZScan(self, img):
        Z = []
        i = j = 0
        while i < 8 and j < 8:
            if j < 7:
                j = j + 1
            else:
                i = i + 1
            while j >= 0 and i < 8:
                Z.append(img[i][j])
                i = i + 1
                j = j - 1
            i = i - 1
            j = j + 1
            if i < 7:
                i = i + 1
            else:
                j = j + 1
            while j < 8 and i >= 0:
                Z.append(img[i][j])
                i = i - 1
                j = j + 1
            i = i + 1
            j = j - 1
        return Z
    def Z2Tab(self, a, table):
        arr = self.RLE(a)
        iter = 0
        i = j = 0
        while i < 8 and j < 8:
            if j < 7:
                j = j + 1
            else:
                i = i + 1
            while j >= 0 and i < 8:
                table[i][j] = arr[iter]
                i = i + 1
                j = j - 1
                iter += 1
            i = i - 1
            j = j + 1
            if i < 7:
                i = i + 1
            else:
                j = j + 1
            while j < 8 and i >= 0:
                table[i][j] = arr[iter]
                i = i - 1
                j = j + 1
                iter += 1
            i = i + 1
            j = j - 1
    def RLC(self, a):
        temp = []
        numof0 = 0
        for num in a:
            if num == 0:
                numof0 += 1
            else:
                temp.append([numof0, num])
                numof0 = 0
        if numof0 != 0:
            temp.append([0, 0])
        return temp
    def RLE(self, a):
        temp = []
        for s in a:
            zero = s[0]
            num = s[1]
            if num == 0:
                for i in range(63 - len(temp)):
                    temp.append(0)
                break
            for i in range(zero):
                temp.append(0)
            temp.append(num)
        return temp

    def DPCM(self, blocks):
        temp = []
        temp.append(blocks[0][0][0])
        for i in range(1, len(blocks)):
            temp.append(blocks[i][0][0] - blocks[i-1][0][0])
        return temp
    def DPCM2(self, arr):
        blocks = []
        for i in range(len(arr)):
            temp = [[0 for i in range(8)] for i in range(8)]
            if i == 0:
                temp[0][0] = arr[0]
            else:
                temp[0][0] = arr[i] + blocks[i-1][0][0]
            blocks.append(temp)
        return blocks

    def toB(self, num):
        num = int(num)
        s = bin(abs(num)).replace('0b', '')
        if num < 0:
            s2 = ''
            for c in s:
                s2 += '0' if c == '1' else '1'
            return s2
        else:
            return s
    def VLI(self, num):
        if num == 0:
            return [0, '']
        s = self.toB(num)
        return [len(s), s]
    def VLIEncoding(self, b):
        s = ''
        if b[0] == '0':
            for c in b:
                s += '0' if c == '1' else '1'
            return -int(s, 2)
        else:
            s += b
            return int(s, 2)
    def AllCompressY(self, DC, arr):
        s = ''
        DC = self.VLI(DC)
        s += self.DC_Y[DC[0]] + DC[1]
        for num in arr:
            runlength = num[0]
            value = num[1]
            temp = self.VLI(value)
            while runlength > 15:
                runlength -= 15
                s += self.AC_Y[(15, 0)]
            s += self.AC_Y[(runlength, temp[0])]
            s += temp[1]
        return s
    def AllCompressUV(self, DC, arr):
        s = ''
        DC = self.VLI(DC)
        s += self.DC_UV[DC[0]] + DC[1]
        for num in arr:
            runlength = num[0]
            value = num[1]
            temp = self.VLI(value)
            while runlength > 15:
                runlength -= 15
                s += self.AC_UV[(15,0)]
            s += self.AC_UV[(runlength, temp[0])]
            s += temp[1]
        return s
    def encoding(self, s, height, width):
        Yheight = (height - 1) // 8 + 1
        Ywidth = (width - 1) // 8 + 1
        UVheight = (height - 1) // 2 // 8 + 1
        UVwidth = (width - 1) // 2 // 8 + 1
        iterHead = 0
        iterTail = 1
        DCY = []
        DCU = []
        DCV = []
        ACY = []
        ACU = []
        ACV = []
        while len(ACY) < Yheight * Ywidth:
            while s[iterHead:iterTail] not in self.DC_Y2.keys() and iterHead<len(s):
                iterTail += 1
            size = self.DC_Y2[s[iterHead:iterTail]]
            iterHead = iterTail
            iterTail += size
            if size == 0:
                amplitude = 0
            else:
                amplitude = self.VLIEncoding(s[iterHead:iterTail])
            DCY.append(amplitude)
            iterHead = iterTail
            iterTail += 1
            ACBlockY = []
            length = 0
            while True:
                runlength = 0
                amplitude = 0
                while True:
                    while s[iterHead:iterTail] not in self.AC_Y2.keys():
                        iterTail += 1
                    t = self.AC_Y2[s[iterHead:iterTail]]
                    if t == (15, 0):
                        runlength += 15
                        iterHead = iterTail
                        iterTail += 1
                    else:
                        runlength += t[0]
                        size = t[1]
                        break
                iterHead = iterTail
                iterTail += size
                if size == 0:
                    amplitude = 0
                else:
                    amplitude = self.VLIEncoding(s[iterHead:iterTail])
                ACBlockY.append([runlength, amplitude])
                iterHead = iterTail
                iterTail += 1
                length += 1
                length += runlength
                if [runlength, amplitude] == [0, 0] or length == 63:
                    break
            ACY.append(ACBlockY)
        while len(ACU) < UVheight * UVwidth:
            while s[iterHead:iterTail] not in self.DC_UV2.keys() and iterHead<len(s):
                iterTail += 1
            size = self.DC_UV2[s[iterHead:iterTail]]
            iterHead = iterTail
            iterTail += size
            if size == 0:
                amplitude = 0
            else:
                amplitude = self.VLIEncoding(s[iterHead:iterTail])
            DCU.append(amplitude)
            iterHead = iterTail
            iterTail += 1
            ACBlockU = []
            while True:
                runlength = 0
                amplitude = 0
                while True:
                    while s[iterHead:iterTail] not in self.AC_UV2.keys():
                        iterTail += 1
                    t = self.AC_UV2[s[iterHead:iterTail]]
                    if t == (15, 0):
                        runlength += 15
                        iterHead = iterTail
                        iterTail += 1
                    else:
                        runlength += t[0]
                        size = t[1]
                        break
                iterHead = iterTail
                iterTail += size
                if size == 0:
                    amplitude = 0
                else:
                    amplitude = self.VLIEncoding(s[iterHead:iterTail])
                ACBlockU.append([runlength, amplitude])
                iterHead = iterTail
                iterTail += 1
                if [runlength, amplitude] == [0, 0]:
                    break
            ACU.append(ACBlockU)
        while len(ACV) < UVheight * UVwidth:
            while s[iterHead:iterTail] not in self.DC_UV2.keys() and iterHead<len(s):
                iterTail += 1
            size = self.DC_UV2[s[iterHead:iterTail]]
            iterHead = iterTail
            iterTail += size
            if size == 0:
                amplitude = 0
            else:
                amplitude = self.VLIEncoding(s[iterHead:iterTail])
            DCV.append(amplitude)
            iterHead = iterTail
            iterTail += 1
            ACBlockV = []
            while True:
                runlength = 0
                amplitude = 0
                while True:
                    while s[iterHead:iterTail] not in self.AC_UV2.keys():
                        iterTail += 1
                    t = self.AC_UV2[s[iterHead:iterTail]]
                    if t == (15, 0):
                        runlength += 15
                        iterHead = iterTail
                        iterTail += 1
                    else:
                        runlength += t[0]
                        size = t[1]
                        break
                iterHead = iterTail
                iterTail += size
                if size == 0:
                    amplitude = 0
                else:
                    amplitude = self.VLIEncoding(s[iterHead:iterTail])
                ACBlockV.append([runlength, amplitude])
                iterHead = iterTail
                iterTail += 1
                if [runlength, amplitude] == [0, 0]:
                    break
            ACV.append(ACBlockV)
        return DCY, DCU, DCV, ACY, ACU, ACV

    def compress(self, img):
        height = img.shape[0]
        width = img.shape[1]
        Y, U, V = self.rgb2yuv(img, img.shape[1], img.shape[0])
        Y = self.fill(Y)
        U = self.fill(U)
        V = self.fill(V)
        blocksY = self.split(Y)
        blocksU = self.split(U)
        blocksV = self.split(V)
        FDCT = []
        Quan = []
        Z = []
        ACnum = []
        for block in blocksY:
            FDCT.append(self.FDCT(block))
            Quan.append(self.quanY(FDCT[-1]))
            Z.append(self.ZScan(Quan[-1]))
            ACnum.append(self.RLC(Z[-1]))
        DCnum = self.DPCM(Quan)
        #print('Y: ')
        Bstr0 = ''
        for i in range(len(ACnum)):
            Bstr0 += self.AllCompressY(DCnum[i], ACnum[i])
        #print(Bstr0)
        #print(len(Bstr0))

        FDCT = []
        Quan = []
        Z = []
        ACnum = []
        for block in blocksU:
            FDCT.append(self.FDCT(block))
            Quan.append(self.quanUV(FDCT[-1]))
            Z.append(self.ZScan(Quan[-1]))
            ACnum.append(self.RLC(Z[-1]))
        DCnum = self.DPCM(Quan)
        #print('U: ')
        Bstr1 = ''
        for i in range(len(ACnum)):
            Bstr1 += self.AllCompressUV(DCnum[i], ACnum[i])
        #print(Bstr1)
        #print(len(Bstr1))

        FDCT = []
        Quan = []
        Z = []
        ACnum = []
        for block in blocksV:
            FDCT.append(self.FDCT(block))
            Quan.append(self.quanUV(FDCT[-1]))
            Z.append(self.ZScan(Quan[-1]))
            ACnum.append(self.RLC(Z[-1]))
        DCnum = self.DPCM(Quan)
        #print('V: ')
        Bstr2 = ''
        for i in range(len(ACnum)):
            Bstr2 += self.AllCompressUV(DCnum[i], ACnum[i])
        #print(Bstr2)
        #print(len(Bstr2))
        s = Bstr0 + Bstr1 + Bstr2
        print(len(s))

        return height, width, s

    def encodings(self, bs, width, height):
        DCY, DCU, DCV, ACY, ACU, ACV = self.encoding(bs, height, width)
        YBlocks = self.DPCM2(DCY)
        UBlocks = self.DPCM2(DCU)
        VBlocks = self.DPCM2(DCV)
        for i in range(len(YBlocks)):
            self.Z2Tab(ACY[i], YBlocks[i])
            YBlocks[i] = self.reY(YBlocks[i])
            YBlocks[i] = self.IDCT(YBlocks[i])
        for i in range(len(UBlocks)):
            self.Z2Tab(ACU[i], UBlocks[i])
            UBlocks[i] = self.reUV(UBlocks[i])
            UBlocks[i] = self.IDCT(UBlocks[i])
        for i in range(len(VBlocks)):
            self.Z2Tab(ACV[i], VBlocks[i])
            VBlocks[i] = self.reUV(VBlocks[i])
            VBlocks[i] = self.IDCT(VBlocks[i])

        Y, U, V = self.merge(YBlocks, UBlocks, VBlocks, height, width)
        img = self.yuv2rgb(Y, U, V, width, height)
        return img



if __name__ == '__main__':
    JPEG = JPEG()
    img = cv2.imread("JPEG/test.jpeg")
    height, width, s = JPEG.compress(img)
    print(height,'*',width,'* 8 * 3 =',height*width*3*8)
    print('压缩比=',(height*width*3*8)/len(s))
    f = open(r'JPEG/code.txt', 'w', encoding='utf-8')
    f.write(s)
    f.close()
    print('图像压缩编码完成！')

    f = open(r'JPEG/code.txt', 'r', encoding='utf-8')
    s = f.read()
    img = JPEG.encodings(s, width, height)
    f.close()
    print('恢复图像完成！')
    cv2.imwrite(r'JPEG/after_encoding.jpg', img)
    cv2.imshow("img after encoding", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
