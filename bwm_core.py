import numpy as np
from numpy.linalg import svd
import copy, cv2
from cv2 import dct, idct
from pywt import dwt2, idwt2
from pool import AutoPool

class WaterMarkCore:
    def __init__(self, password_img=1, mode='common', processes=None, d1 = 9, d2 = 7, fast_mode = False, n = 3):
        self.n = n
        self.block_shape = np.array([self.n, self.n])#([4, 4])
        self.password_img = password_img
        self.d1, self.d2 = d1, d2  # интервалы квантования
        # 36, 20  # d1/d2 越大鲁棒性越强,但输出图片的失真越大
        # Чем больше размер, тем выше стойкость, но тем больше искажается выходное изображение.
        # init data
        self.img, self.img_YUV = None, None  # self.img 是原图，self.img_YUV 对像素做了加白偶数化 = исходное изображение, self.img_YUV отбеливает и выравнивает пиксели
        self.ca, self.hvd, = [np.array([])] * 3, [np.array([])] * 3  # 每个通道 dct 的结果 = Результаты для: dct на канал
        self.ca_block = [np.array([])] * 3  # 每个 channel 存一个四维 array，代表四维分块后的结果 = Каждый канал хранит четырехмерный массив, представляющий результат четырехмерной разбивки.
        self.ca_part = [np.array([])] * 3  # 四维分块后，有时因不整除而少一部分，self.ca_part 是少这一部分的 self.ca = После четырехмерной разбивки иногда часть отсутствует из-за нецелочисленного
                                                                                                                    # деления. self.ca_part - это часть self.ca, которая отсутствует.
        self.wm_size, self.block_num = 0, 0  # 水印的长度，原图片可插入信息的个数 = Длина водяного знака, количество фрагментов информации, которые могут быть вставлены в исходное изображение
        self.pool = AutoPool(mode=mode, processes=processes)
        self.fast_mode = fast_mode
        self.alpha = None  # 用于处理透明图 = Используется для обработки диаграмм прозрачности

    def init_block_index(self):
        self.block_num = self.ca_block_shape[0] * self.ca_block_shape[1]
        assert self.wm_size < self.block_num, IndexError(
            '最多可嵌入{}kb信息，多于水印的{}kb信息，溢出'.format(self.block_num / 1000, self.wm_size / 1000))
        # До {} кб встроенной информации, более {} кб информации с водяными знаками, переполнение
        # self.part_shape 是取整后的ca二维大小,用于嵌入时忽略右边和下面对不齐的细条部分。
        # это округленный двумерный размер ca, который используется для игнорирования смещения элементов справа и снизу при встраивании.
        self.part_shape = self.ca_block_shape[:2] * self.block_shape
        self.block_index = [(i, j) for i in range(self.ca_block_shape[0]) for j in range(self.ca_block_shape[1])]

    def read_img_arr(self, img):
        # 处理透明图 = Обработка прозрачных изображений
        self.alpha = None
        if img.shape[2] == 4:
            if img[:, :, 3].min() < 255:
                self.alpha = img[:, :, 3]
                img = img[:, :, :3]
        # 读入图片->YUV化->加白边使像素变偶数->四维分块 = Считывание изображения -> YUVise -> добавление белой границы, чтобы сделать пиксели равномерными -> 4D чанкинг
        self.img = img.astype(np.float32)
        self.img_shape = self.img.shape[:2]
        # 如果不是偶数，那么补上白边，Y（明亮度）UV（颜色）= Если это не четное число, то заполните белые края, Y (яркость) UV (цвет)
        self.img_YUV = cv2.copyMakeBorder(cv2.cvtColor(self.img, cv2.COLOR_BGR2YUV),
                                          0, self.img.shape[0] % 2, 0, self.img.shape[1] % 2,
                                          cv2.BORDER_CONSTANT, value=(0, 0, 0))
        self.ca_shape = [(i + 1) // 2 for i in self.img_shape]
        self.ca_block_shape = (self.ca_shape[0] // self.block_shape[0], self.ca_shape[1] // self.block_shape[1],
                               self.block_shape[0], self.block_shape[1])
        strides = 4 * np.array([self.ca_shape[1] * self.block_shape[0], self.block_shape[1], self.ca_shape[1], 1])
        for channel in range(3):
            self.ca[channel], self.hvd[channel] = dwt2(self.img_YUV[:, :, channel], 'haar')
            # 转为4维度 = Переход к 4 измерениям
            self.ca_block[channel] = np.lib.stride_tricks.as_strided(self.ca[channel].astype(np.float32),
                                                                     self.ca_block_shape, strides)

    def read_wm(self, wm_bit):
        self.wm_bit = wm_bit
        self.wm_size = wm_bit.size

    def block_add_wm(self, arg):
        if self.fast_mode:
            return self.block_add_wm_fast(arg)
        else:
            return self.block_add_wm_slow(arg)

    def block_add_wm_slow(self, arg):
        # block, shuffler, i = arg
        block, _ , i = arg
        # dct->(flatten->加密->逆flatten)->svd->打水印->逆svd->(flatten->解密->逆flatten)->逆dct
        # dct->(flatten->encrypt->inverse flatten)->svd->watermark->inverse svd->(flatten->decrypt->inverse flatten)->inverse dct
        wm_1 = self.wm_bit[i % self.wm_size]
        block_dct = dct(block)
        # 加密（打乱顺序）= Шифрование (не по порядку)
        # block_dct_shuffled = block_dct.flatten()[shuffler].reshape(self.block_shape)
        block_dct_shuffled = block_dct.reshape(self.block_shape)
        u, s, v = svd(block_dct_shuffled)
        s[0] = (s[0] // self.d1 + 1 / 4 + 1 / 2 * wm_1) * self.d1
        if self.d2:
            s[1] = (s[1] // self.d2 + 1 / 4 + 1 / 2 * wm_1) * self.d2
        block_dct_flatten = np.dot(u, np.dot(np.diag(s), v))#.flatten()
        # block_dct_flatten[shuffler] = block_dct_flatten.copy()
        block_dct_flatten = block_dct_flatten
        return idct(block_dct_flatten.reshape(self.block_shape))

    def block_add_wm_fast(self, arg):
        # dct->svd->打水印->逆svd->逆dct = dct->svd->watermark->reverse svd->reverse dct
        # block, shuffler, i = arg
        block, _ , i = arg
        wm_1 = self.wm_bit[i % self.wm_size] # "мигалка" = Бу́лева фу́нкция (или логи́ческая функция)
        # print(f'wm_1 = {wm_1}') # True или False

        u, s, v = svd(dct(block))

        # s[0] = (s[0] // self.d1 + 0.25 + 0.5 * wm_1) * self.d1
        # print(f's[0] before = {s[0]}')
        # s[0] = (s[0] // self.d1 + 1/4 + 1/16 + 1/32 + 1/2 * wm_1) * self.d1 # работает, но хуже

        # from math import ceil
        # s[0] = ceil(s[0]) + (1/4 + 1/16 + 1/64 + 1/2 * wm_1) * self.d1 # некорректное извлечение

        # print(f's[0] after = {s[0]}')
        s[0] = (s[0] // self.d1 + 1/4 + 1/2 * wm_1) * self.d1  # 1/2*True = 1/2*1 = 1/2; 1/2*False = 1/2*0 = 0
        # s[0] // d1 = целой части от деления s[0] / d1, if s[0] > d1, else s[0] // d1 = 0
        # print(f's[0] = {s[0]}, d1 = {self.d1}, s[0] // self.d1 = {s[0] // self.d1},\
        #  \ns[0] // self.d1 + 1/4 + 1/2 * wm_1 = {s[0] // self.d1 + 1/4 + 1/2 * wm_1},\n \
        #  (s[0] // self.d1 * self.d1 + 1/4* self.d1 + 1/2 * wm_1* self.d1 = {s[0] // self.d1 * self.d1 + 1/4 * self.d1+ 1/2 * wm_1 * self.d1}, \n \
        #  s[0] // self.d1 * self.d1 = {s[0] // self.d1 * self.d1},\n \
        #  1/2 * wm_1 * self.d1 = {1/2 * wm_1 * self.d1}')
        return idct(np.dot(u, np.dot(np.diag(s), v)))

    def embed(self):
        self.init_block_index()
        embed_ca = copy.deepcopy(self.ca)
        embed_YUV = [np.array([])] * 3
        self.idx_shuffle = random_strategy1(self.password_img, self.block_num,
                                            self.block_shape[0] * self.block_shape[1])
        for channel in range(3):
            tmp = self.pool.map(self.block_add_wm,
                                [(self.ca_block[channel][self.block_index[i]], self.idx_shuffle[i], i)
                                 for i in range(self.block_num)])
            for i in range(self.block_num):
                self.ca_block[channel][self.block_index[i]] = tmp[i]
            # 4维分块变回2维 = Четырехмерное преобразование в двухмерное
            self.ca_part[channel] = np.concatenate(np.concatenate(self.ca_block[channel], 1), 1)
            # 4维分块时右边和下边不能整除的长条保留，其余是主体部分，换成 embed 之后的频域的数据
            # При 4-мерном разбиении правая и нижняя часть длинной полосы, которая не является делимой, сохраняется,
            # а остальное - это основная часть данных, которая после встраивания преобразуется в частотную область.
            embed_ca[channel][:self.part_shape[0], :self.part_shape[1]] = self.ca_part[channel]
            # 逆变换回去 = инверсия
            embed_YUV[channel] = idwt2((embed_ca[channel], self.hvd[channel]), "haar")
        # 合并3通道 = Объединить 3 канала
        embed_img_YUV = np.stack(embed_YUV, axis=2)
        # 之前如果不是2的整数，增加了白边，这里去除掉
        # Ранее, если оно не было целым числом 2, оно добавляло белую рамку, которая здесь удалена
        embed_img_YUV = embed_img_YUV[:self.img_shape[0], :self.img_shape[1]]
        embed_img = cv2.cvtColor(embed_img_YUV, cv2.COLOR_YUV2BGR)
        embed_img = np.clip(embed_img, a_min=0, a_max=255)
        if self.alpha is not None:
            embed_img = cv2.merge([embed_img.astype(np.uint8), self.alpha])
        return embed_img

    def block_get_wm(self, args):
        if self.fast_mode:
            return self.block_get_wm_fast(args)
        else:
            return self.block_get_wm_slow(args)

    def block_get_wm_slow(self, args):
        # block, shuffler = args
        block, _ = args
        # dct->flatten->加密->逆flatten->svd->解水印 = dct->flatten->encrypt->inverse flatten->svd->unwatermark
        
        # block_dct_shuffled = dct(block).flatten()[shuffler].reshape(self.block_shape)
        block_dct_shuffled = dct(block).reshape(self.block_shape)

        u, s, v = svd(block_dct_shuffled)
        wm = (s[0] % self.d1 > self.d1 / 2) * 1
        if self.d2:
            tmp = (s[1] % self.d2 > self.d2 / 2) * 1
            wm = (wm * 3 + tmp * 1) / 4
        return wm

    def block_get_wm_fast(self, args):
        # block, shuffler = args
        block, _ = args
        # dct->svd->解水印 = dct->svd->unwatermark
        u, s, v = svd(dct(block))
        wm = (s[0] % self.d1 > self.d1 / 2)  # wm = 0 or wm = 1
        # print(f's[0] = {s[0]}, wm = {wm}') 
        return wm

    def extract_raw(self, img):
        # 每个分块提取 1 bit 信息 = Извлечение 1 бита информации из каждого чанка
        self.read_img_arr(img=img)
        self.init_block_index()
        wm_block_bit = np.zeros(shape=(3, self.block_num))  # 3个channel，length 个分块提取的水印，全都记录下来 = 3 канала,
                                                            # длина водяных знаков извлекается кусками, все записано.
        self.idx_shuffle = random_strategy1(seed=self.password_img,
                                            size=self.block_num,
                                            block_shape=self.block_shape[0] * self.block_shape[1],  # 16
                                            )
        for channel in range(3):
            wm_block_bit[channel, :] = self.pool.map(self.block_get_wm,
                                                     [(self.ca_block[channel][self.block_index[i]], self.idx_shuffle[i])
                                                      for i in range(self.block_num)])
        return wm_block_bit

    def extract_avg(self, wm_block_bit):
        # 对循环嵌入+3个 channel 求平均 = Усреднение по циклическому вкраплению + 3 канала
        wm_avg = np.zeros(shape=self.wm_size)
        for i in range(self.wm_size):
            wm_avg[i] = wm_block_bit[:, i::self.wm_size].mean()
        return wm_avg

    def extract(self, img, wm_shape):
        self.wm_size = np.array(wm_shape).prod()
        # 提取每个分块埋入的 bit：= Извлеките бит, содержащийся в каждом фрагменте
        wm_block_bit = self.extract_raw(img=img)
        # 做平均：= найти оптимальный баланс
        wm_avg = self.extract_avg(wm_block_bit)
        return wm_avg

    def extract_with_kmeans(self, img, wm_shape):
        wm_avg = self.extract(img=img, wm_shape=wm_shape)
        return one_dim_kmeans(wm_avg)

def one_dim_kmeans(inputs):
    threshold = 0
    e_tol = 10 ** (-6)
    center = [inputs.min(), inputs.max()]  # 1. 初始化中心点 = Инициализация центральной точки
    for i in range(300):
        threshold = (center[0] + center[1]) / 2
        is_class01 = inputs > threshold  # 2. 检查所有点与这k个点之间的距离，每个点归类到最近的中心
        # Проверьте расстояния между всеми точками и этими k точками, каждая из которых классифицирована до ближайшего центра
        center = [inputs[~is_class01].mean(), inputs[is_class01].mean()]  # 3. 重新找中心点 = Заново открывая центральную точку
        if np.abs((center[0] + center[1]) / 2 - threshold) < e_tol:  # 4. 停止条件 = условие остановки
            threshold = (center[0] + center[1]) / 2
            break
    is_class01 = inputs > threshold
    return is_class01

def random_strategy1(seed, size, block_shape):
    return np.random.RandomState(seed) \
        .random(size=(size, block_shape)) \
        .argsort(axis=1)
