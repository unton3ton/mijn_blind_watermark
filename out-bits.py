import os, glob, bitarray
from PIL import Image
from blind_watermark import WaterMark


def extract_watermark(img_name, lenght_watermark, passwordwm=1, d1 = 9, d2 = 7, fast_mode = True, n = 3):
    bwm1 = WaterMark(password_img=1, password_wm=passwordwm, mode='common', d1 = d1, d2 = d2, fast_mode = fast_mode, n = n)
    wm_extract = bwm1.extract(img_name, wm_shape=lenght_watermark, mode='bit')
    return wm_extract


with open('photos/results-bits.txt','w', encoding="utf-8") as f:

    images = glob.glob("photos/Embedded-bits/*.jpg") 
    # images = glob.glob("photos/fromTelegram/*.jpg")

    d1 = d2 = 6

    for name in images:
        try:
            # don't work:
            # wm_bit = bitarray.bitarray(extract_watermark(name, lenght_watermark=2, passwordwm=123456789, d1 = d1, d2 = d2))
            
            wm_bit = extract_watermark(name, lenght_watermark=12, passwordwm=123456789, d1 = d1, d2 = d2)
            wm_text = wm_bit.tobytes().decode('utf-8')

            print(f'for {name[:-4]} wm_bit = \n {wm_bit}\n')
            print(f'for {name[:-4]} wm_text = \n {wm_text}\n')
            f.write(f'\nfor {name[:-4]} wm_bit = \n {wm_bit}\n wm_text = \n {wm_text}\n')
        except ValueError:
            print(f"ValueError: non-hexadecimal number found in fromhex() arg at position 127 for file = {name[:-4]}\n")
            f.write(f"\nValueError: non-hexadecimal number found in fromhex() arg at position 127 for file = {name[:-4]}\n")
        finally:
            continue