## Blind watermark based on DWT-DCT-SVD.

# install
```bash
pip install blind-watermark
```

For the current developer version:
```bach
git clone git@github.com:unton3ton/mijn_blind_watermark.git
cd mijn_blind_watermark
pip install .
```

# How to use


## Use in Python


Embed watermark:
```python
import os 
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
```

Extract watermark:
```python
import os 
from PIL import Image
from blind_watermark import WaterMark
import glob

def extract_watermark(img_name, lenght_watermark, passwordwm=1, d1 = 9, d2 = 7, fast_mode = True, n = 3):
    bwm1 = WaterMark(password_img=1, password_wm=passwordwm, mode='common', d1 = d1, d2 = d2, fast_mode = fast_mode, n = n)
    wm_extract = bwm1.extract(img_name, wm_shape=lenght_watermark, mode='str')
    return wm_extract


with open('photos/results.txt','w', encoding="utf-8") as f:

    images = glob.glob("photos/Embedded/*.jpg") 
    # images = glob.glob("photos/fromTelegram/*.jpg")

    d1 = d2 = 6

    for name in images:
        try:
            print(f'for {name[:-4]} = \n {extract_watermark(name, lenght_watermark=1039, passwordwm=123456789, d1 = d1, d2 = d2)}\n')
            f.write(f'\nfor {name[:-4]} = \n {extract_watermark(name, lenght_watermark=1039, passwordwm=123456789, d1 = 6, d2 = 6)}\n')
        except ValueError:
            print(f"ValueError: non-hexadecimal number found in fromhex() arg at position 127 for file = {name[:-4]}\n")
            f.write(f"\nValueError: non-hexadecimal number found in fromhex() arg at position 127 for file = {name[:-4]}\n")
        finally:
            continue
```


### embed bits

embed watermark:
```python
import bitarray

wm = '012345678910'

times = 1
print(wm*times)

ba = bitarray.bitarray()
ba.frombytes(wm.encode('utf-8'))
print(ba) # bitarray('001100000011000100110010001100110011010000110101001101100011011100111000001110010011000100110000')
```


Extract watermark:
```python
wm_bit = extract_watermark(name, lenght_watermark=12, passwordwm=123456789, d1 = d1, d2 = d2)
wm_text = wm_bit.tobytes().decode('utf-8')
```


## Sources

* [Ma3shka](https://github.com/unton3ton/Ma3shka)
* [Как уменьшить количество измерений и извлечь из этого пользу](https://habr.com/ru/articles/275273/)
* [Singular Value Decomposition (SVD)](https://www.geeksforgeeks.org/singular-value-decomposition-svd/)
* [bitarray 2.9.2](https://pypi.org/project/bitarray/)
* [Convert string to list of bits and viceversa](https://stackoverflow.com/questions/10237926/convert-string-to-list-of-bits-and-viceversa)
* [in-search-of-incredible-difference](https://github.com/tonypithony/in-search-of-incredible-difference)
