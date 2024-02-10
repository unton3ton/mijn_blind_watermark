# https://github.com/guofei9987/blind_watermark/tree/master/blind_watermark
## the right terminale (base): >> python in.py && python out.py

# pip install opencv-python
# pip install PyWavelets

import os #, time
from blind_watermark import WaterMark


def lenght_watermark(img_name, watermark, passwordwm=1):
    bwm1 = WaterMark(password_img=1, password_wm=passwordwm) # mode='common' vs mode='multithreading'
    bwm1.read_img(f'{path}/{img_name}')
    bwm1.read_wm(watermark, mode='str')
    len_wm = len(bwm1.wm_bit)
    return len_wm

def embed_watermark(img_name, watermark, passwordwm=1, compression_ratio=100, d1 = 9, d2 = 7, fast_mode = True, n = 3):
    bwm1 = WaterMark(password_img=1, password_wm=passwordwm, mode='common', d1 = d1, d2 = d2, fast_mode = fast_mode, n = n) 
    bwm1.read_img(f'{path}/{img_name}')
    bwm1.read_wm(watermark, mode='str')
    bwm1.embed(f'{path}/Embedded/{img_name[:-4]}_compression_{compression_ratio}_d1_{d1}_d2_{d2}.jpg', compression_ratio=compression_ratio)


path = "photos/"

password_wm = 123456789

# wm = '99'
with open('original.txt','r', encoding="utf-8") as f:
    wm = f.read()

times = 2
print(wm*times)

d1 = d2 = 6

for i in range(1,7):
    name = f'{i}.jpg'
    print(f'lwm = {lenght_watermark(name, wm*times)}') # 519 for times = 1; 1559 for times = 3
    embed_watermark(name, wm*times, password_wm, d1 = d1, d2 = d2)

# i = 4
# name = f'{i}.jpg'
# lwm = lenght_watermark(name, wm)
# print(f'lwm = {lwm}') # 511
# embed_watermark(name, wm, password_wm, d1 = 6, d2 = 6)

# N = 9
# for i in range(1,N):
#     name = f'{i}.jpg'
#     print(f'{i} lwm = {lenght_watermark(name, wm)}') # 511
#     embed_watermark(name, wm, password_wm)
# print()

# # N = 9
# # tic = time.perf_counter()
# for i in range(4,5):
#     for d1 in range(7,9,1):
#         for d2 in range(7,9,1):
#             for compression in reversed(range(95,105,5)): # от 100 до 0 с шагом 10
#                 name = f'{i}.jpg'
#                 print(f'{i} lwm = {lenght_watermark(name, wm)}, compression = {compression}') # 511
#                 embed_watermark(name, wm, password_wm, compression, d1 = d1, d2 = d2)
# # toc = time.perf_counter()
# # print(f"Внедрение multithreading заняло {toc - tic:0.1f} секунд")
# print()