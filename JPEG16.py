import numpy as np
import os
from PIL import Image

class KJPEG:
    def __init__(self, scale, WR, WG, WB, TY, TU, TV):
        # 初始化DCT变换的A矩阵，https://blog.csdn.net/ahafg/article/details/48808443
        self.__scale = scale
        self.__dctA = np.zeros(shape=(scale, scale))
        for i in range(scale):
            c = 0
            if i == 0:
                c = np.sqrt(1 / scale)
            else:
                c = np.sqrt(2 / scale)
            for j in range(scale):
                self.__dctA[i, j] = c * np.cos(np.pi * i * (2 * j + 1) / (2 * scale))
        # 参数
        self.__WR = WR
        self.__WG = WG
        self.__WB = WB
        # 亮度量化矩阵
        self.__lq = TY
        # 色度量化矩阵
        self.__cq1 = TU
        self.__cq2 = TV
        # 标记矩阵类型，lt是亮度矩阵，ct是色度矩阵
        self.__lt = 0
        self.__ct1 = 1
        self.__ct2 = 2
        # https://my.oschina.net/tigerBin/blog/1083549
        # Zig编码表
        # self.__zig = np.array([
        #     0  ,  1, 16, 32, 17,  2,  3, 18, 33, 48, 64, 49, 34, 19,  4,  5,
        #     20 , 35, 50, 65, 80, 96, 81, 66, 51, 36, 21,  6,  7, 22, 37, 52,
        #     67 , 82, 97,112,128,113, 98, 83, 68, 53, 38, 23,  8,  9, 24, 39,
        #     54 , 69, 84, 99,114,129,144,160,145,130,115,100, 85, 70, 55, 40,
        #     25 , 10, 11, 26, 41, 56, 71, 86,101,116,131,146,161,176,192,177,
        #     162,147,132,117,102, 87, 72, 57, 42, 27, 12, 13, 28, 43, 58, 73,
        #     88 ,103,118,133,148,163,178,193,208,224,209,194,179,164,149,134,
        #     119,104, 89, 74, 59, 44, 29, 14, 15, 30, 45, 60, 75, 90,105,120,
        #     135,150,165,180,195,210,225,240,241,226,211,196,181,166,151,136,
        #     121,106, 91, 76, 61, 46, 31, 47, 62, 77, 92,107,122,137,152,167,
        #     182,197,212,227,242,243,228,213,198,183,168,153,138,123,108, 93,
        #     78 , 63, 79, 94,109,124,139,154,169,184,199,214,229,244,245,230,
        #     215,200,185,170,155,140,125,110, 95,111,126,141,156,171,186,201,
        #     216,231,246,247,232,217,202,187,172,157,142,127,143,158,173,188,
        #     203,218,233,248,249,234,219,204,189,174,159,175,190,205,220,235,
        #     250,251,236,221,206,191,207,222,237,252,253,238,223,239,254,255
        # ])
        # # Zag编码表
        # self.__zag = np.array([
        #     0  ,  1,  5,  6, 14, 15, 27, 28, 44, 45, 65, 66, 90, 91,119,120,
        #     2  ,  4,  7, 13, 16, 26, 29, 43, 46, 64, 67, 89, 92,118,121,150,
        #     3  ,  8, 12, 17, 25, 30, 42, 47, 63, 68, 88, 93,117,122,149,151,
        #     9  , 11, 18, 24, 31, 41, 48, 62, 69, 87, 94,116,123,148,152,177,
        #     10 , 19, 23, 32, 40, 49, 61, 70, 86, 95,115,124,147,153,176,178,
        #     20 , 22, 33, 39, 50, 60, 71, 85, 96,114,125,146,154,175,179,200,
        #     21 , 34, 38, 51, 59, 72, 84, 97,113,126,145,155,174,180,199,201,
        #     35 , 37, 52, 58, 73, 83, 98,112,127,144,156,173,181,198,202,219,
        #     36 , 53, 57, 74, 82, 99,111,128,143,157,172,182,197,203,218,220,
        #     54 , 56, 75, 81,100,110,129,142,158,171,183,196,204,217,221,234,
        #     55 , 76, 80,101,109,130,141,159,170,184,195,205,216,222,233,235,
        #     77 , 79,102,108,131,140,160,169,185,194,206,215,223,232,236,245,
        #     78 ,103,107,132,139,161,168,186,193,207,214,224,231,237,244,246,
        #     104,106,133,138,162,167,187,192,208,213,225,230,238,243,247,252,
        #     105,134,137,163,166,188,191,209,212,226,229,239,242,248,251,253,
        #     135,136,164,165,189,190,210,211,227,228,240,241,249,250,254,255
        # ])

    def __Rgb2Yuv(self, r, g, b):
        # 从图像获取YUV矩阵
        y = self.__WR * r + self.__WG * g + self.__WB * b
        u = 0.5 *(((-self.__WR)/(1-self.__WB)) * r + ((-self.__WG)/(1-self.__WB)) * g + b) + 128
        v = 0.5 *(((-self.__WB)/(1-self.__WR)) * b + ((-self.__WG)/(1-self.__WR)) * g + r) + 128
        return y, u, v

    def __Fill(self, matrix):
        # 图片的长宽都需要满足是16的倍数（采样长宽会缩小1/2和取块长宽会缩小1/8）
        # 图像压缩三种取样方式4:4:4、4:2:2、4:2:0
        scale = self.__scale
        fh, fw = 0, 0
        if self.height % (2*scale) != 0:
            fh = (2*scale) - self.height % (2*scale)
        if self.width % (2*scale) != 0:
            fw = (2*scale) - self.width % (2*scale)
        res = np.pad(matrix, ((0, fh), (0, fw)), 'constant',
                             constant_values=(0, 0))
        return res

    def __Encode(self, matrix, tag):
        scale = self.__scale
        # 先对矩阵进行填充
        matrix = self.__Fill(matrix)
        # 将图像矩阵切割成8*8小块
        height, width = matrix.shape
        # 减少for循环语句，利用numpy的自带函数来提升算法效率
        # 参考吴恩达的公开课视频，numpy的函数自带并行处理，不用像for循环一样串行处理
        shape = (height // scale, width // scale, scale, scale)
        strides = matrix.itemsize * np.array([width * scale, scale, width, 1])
        blocks = np.lib.stride_tricks.as_strided(matrix, shape=shape, strides=strides)
        res = []
        for i in range(height // scale):
            for j in range(width // scale):
                res.append(self.__Quantize(self.__Dct(blocks[i, j]).reshape(scale*scale), tag))
        return res

    def __Dct(self, block):
        # DCT变换
        res = np.dot(self.__dctA, block)
        res = np.dot(res, np.transpose(self.__dctA))
        return res

    def __Quantize(self, block, tag):
        res = block
        if tag == self.__lt:
            res = np.round(res / self.__lq)
        elif tag == self.__ct1:
            res = np.round(res / self.__cq1)
        elif tag == self.__ct2:
            res = np.round(res / self.__cq2)
        return res

    # def __Zig(self, blocks):
    #     ty = np.array(blocks)
    #     tz = np.zeros(ty.shape)
    #     for i in range(len(self.__zig)):
    #         tz[:, i] = ty[:, self.__zig[i]]
    #     tz = tz.reshape(tz.shape[0] * tz.shape[1])
    #     return tz.tolist()

    # def __Rle(self, blist):
    #     res = []
    #     cnt = 0
    #     for i in range(len(blist)):
    #         if blist[i] != 0:
    #             res.append(cnt)
    #             res.append(int(blist[i]))
    #             cnt = 0
    #         elif cnt == 15:
    #             res.append(cnt)
    #             res.append(int(blist[i]))
    #             cnt = 0
    #         else:
    #             cnt += 1
    #     # 末尾全是0的情况
    #     if cnt != 0:
    #         res.append(cnt - 1)
    #         res.append(0)
    #     return res

    def Compress(self, filename):
        # 根据路径image_path读取图片，并存储为RGB矩阵
        image = Image.open(filename)
        # 获取图片宽度width和高度height
        self.width, self.height = image.size
        image = image.convert('RGB')
        image = np.asarray(image)
        r = image[:, :, 0]
        g = image[:, :, 1]
        b = image[:, :, 2]
         # 将图像RGB转YUV
        y, u, v = self.__Rgb2Yuv(r, g, b)
        # 对图像矩阵进行量化
        y_blocks = self.__Encode(y, self.__lt)
        u_blocks = self.__Encode(u, self.__ct1)
        v_blocks = self.__Encode(v, self.__ct2)
        # # 对图像小块进行Zig编码和RLE编码
        # y_code = self.__Rle(self.__Zig(y_blocks))
        # u_code = self.__Rle(self.__Zig(u_blocks))
        # v_code = self.__Rle(self.__Zig(v_blocks))
        # return y_code, u_code, v_code
        return y_blocks, u_blocks, v_blocks


    def __IDct(self, block):
        # IDCT
        res = np.dot(np.transpose(self.__dctA), block)
        res = np.dot(res, self.__dctA)
        return res

    def __IQuantize(self, block, tag):
        res = block
        if tag == self.__lt:
            res *= self.__lq
        elif tag == self.__ct1:
            res *= self.__cq1
        elif tag == self.__ct2:
            res *= self.__cq2
        return res

    def __IFill(self, matrix):
        matrix = matrix[:self.height, :self.width]
        return matrix

    def __Decode(self, blocks, tag):
        scale = self.__scale
        tlist = []
        for b in blocks:
            b = np.array(b)
            tlist.append(self.__IDct(self.__IQuantize(b, tag).reshape(scale ,scale)))
        height_fill, width_fill = self.height, self.width
        if height_fill % (2*scale) != 0:
            height_fill += (2*scale) - height_fill % (2*scale)
        if width_fill % (2*scale) != 0:
            width_fill += (2*scale) - width_fill % (2*scale)
        rlist = []
        for hi in range(height_fill // scale):
            start = hi * width_fill // scale
            rlist.append(np.hstack(tuple(tlist[start: start + (width_fill // scale)])))
        matrix = np.vstack(tuple(rlist))
        res = self.__IFill(matrix)
        return res

    # def __Zag(self, dcode):
    #     dcode = np.array(dcode).reshape((len(dcode)//256, 256))
    #     tz = np.zeros(dcode.shape)
    #     for i in range(len(self.__zag)):
    #         tz[:, i] = dcode[:, self.__zag[i]]
    #     rlist = tz.tolist()
    #     return rlist

    # def __IRle(self, dcode):
    #     rlist = []
    #     for i in range(len(dcode)):
    #         if i % 2 == 0:
    #             rlist += [0] * dcode[i]
    #         else:
    #             rlist.append(dcode[i])
    #     return rlist

    def Decompress(self, y_blocks, u_blocks, v_blocks):
        # y_blocks = self.__Zag(self.__IRle(y_dcode))
        # u_blocks = self.__Zag(self.__IRle(u_dcode))
        # v_blocks = self.__Zag(self.__IRle(v_dcode))
        y = self.__Decode(y_blocks, self.__lt)
        u = self.__Decode(u_blocks, self.__ct1)
        v = self.__Decode(v_blocks, self.__ct2)
        r = (y + 2*(1-self.__WR) * (v - 128))
        g = (y - (2*(1-self.__WB)*self.__WB/self.__WG) * (u - 128) - (2*(1-self.__WR)*self.__WR/self.__WG) * (v - 128))
        b = (y + 2*(1-self.__WB) * (u - 128))
        r = Image.fromarray(r).convert('L')
        g = Image.fromarray(g).convert('L')
        b = Image.fromarray(b).convert('L')
        image = Image.merge("RGB", (r, g, b))
        return image

if __name__ == '__main__':
    
    WR = 0.299
    WG = 0.587
    WB = 0.114
    TY = np.array([
        16, 11, 10, 16, 24, 40, 51, 61,
        12, 12, 14, 19, 26, 58, 60, 55,
        14, 13, 16, 24, 40, 57, 69, 56,
        14, 17, 22, 29, 51, 87, 80, 62,
        18, 22, 37, 56, 68, 109, 103, 77,
        24, 35, 55, 64, 81, 104, 113, 92,
        49, 64, 78, 87, 103, 121, 120, 101,
        72, 92, 95, 98, 112, 100, 103, 99,
    ])
    # 色度量化矩阵
    TU = np.array([
        17, 18, 24, 47, 99, 99, 99, 99,
        18, 21, 26, 66, 99, 99, 99, 99,
        24, 26, 56, 99, 99, 99, 99, 99,
        47, 66, 99, 99, 99, 99, 99, 99,
        99, 99, 99, 99, 99, 99, 99, 99,
        99, 99, 99, 99, 99, 99, 99, 99,
        99, 99, 99, 99, 99, 99, 99, 99,
        99, 99, 99, 99, 99, 99, 99, 99,
    ])
    TV = np.array([
        17, 18, 24, 47, 99, 99, 99, 99,
        18, 21, 26, 66, 99, 99, 99, 99,
        24, 26, 56, 99, 99, 99, 99, 99,
        47, 66, 99, 99, 99, 99, 99, 99,
        99, 99, 99, 99, 99, 99, 99, 99,
        99, 99, 99, 99, 99, 99, 99, 99,
        99, 99, 99, 99, 99, 99, 99, 99,
        99, 99, 99, 99, 99, 99, 99, 99,
    ])
    kjpeg = KJPEG(WR, WG, WB, TY, TU, TV)
    y_code, u_code, v_code = kjpeg.Compress("../photograph.jpg")
    img = kjpeg.Decompress(y_code, u_code, v_code)
    img.save("result_new.jpg", "jpeg")