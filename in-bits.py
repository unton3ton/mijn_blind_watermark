# https://github.com/guofei9987/blind_watermark/tree/master/blind_watermark
## the right terminale (base): >> python in.py && python out.py

# pip install opencv-python
# pip install PyWavelets

# conda activate TesseracT

import os, bitarray # pip install bitarray
from blind_watermark import WaterMark


def lenght_watermark(img_name, watermark, passwordwm=1):
    bwm1 = WaterMark(password_img=1, password_wm=passwordwm) # mode='common' vs mode='multithreading'
    bwm1.read_img(f'{path}/{img_name}')
    bwm1.read_wm(watermark, mode='bit')
    len_wm = len(bwm1.wm_bit)
    return len_wm

def embed_watermark(img_name, watermark, passwordwm=1, compression_ratio=100, d1 = 9, d2 = 7, fast_mode = True, n = 3):
    bwm1 = WaterMark(password_img=1, password_wm=passwordwm, mode='common', d1 = d1, d2 = d2, fast_mode = fast_mode, n = n) 
    bwm1.read_img(f'{path}/{img_name}')
    bwm1.read_wm(watermark, mode='bit')
    bwm1.embed(f'{path}/Embedded-bits/{img_name[:-4]}_compression_{compression_ratio}_d1_{d1}_d2_{d2}.jpg', compression_ratio=compression_ratio)


path = "photos/"

password_wm = 123456789

wm = '012345678910'
# with open('original.txt','r', encoding="utf-8") as f:
#     wm = f.read()

times = 1
print(wm*times)


ba = bitarray.bitarray()
ba.frombytes(wm.encode('utf-8'))
print(ba) # bitarray('001100000011000100110010001100110011010000110101001101100011011100111000001110010011000100110000')


d1 = d2 = 6

for i in range(1,7):
    name = f'{i}.jpg'
    print(f'lwm = {lenght_watermark(name, ba)}') # 519 for times = 1; 1559 for times = 3
    embed_watermark(name, ba, password_wm, d1 = d1, d2 = d2)
