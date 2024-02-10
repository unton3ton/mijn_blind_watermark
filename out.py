import os #, time
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


    # os.chdir("photos/Embedded") 
    # os.chdir("fromTelegram")
    # os.chdir("photos/Embedded/Compressed")


    # name = "4_compression_100_d1_9_d2_7.jpg"
    # extract_wm = extract_watermark(name, lenght_watermark=262, passwordwm=123456789, d1 = 9, d2 = 7)
    # print(extract_wm)
    # f.write(extract_wm)
    

    # N = 9
    # for i in range(1,N):
    #     name = f'embedded_text_{i}.jpg'
    #     print(i, extract_watermark(name, lenght_watermark=511, passwordwm=123456789), '\n')


    # print("Empty test: ", extract_watermark('empty.jpg', lenght_watermark=511, passwordwm=123456789))


    # # N = 10
    # # tic = time.perf_counter()
    # for i in range(4,5):
    #     for d1 in range(7,9,1):
    #         for d2 in range(7,9,1):
    #             for compression in reversed(range(95,105,5)):
    #                 name = f'{i}_compression_{compression}_d1_{d1}_d2_{d2}.jpg'
    #                 try:
    #                     print(f'{i} {extract_watermark(name, lenght_watermark=511, passwordwm=123456789, d1 = d1, d2 = d2)}, compression={compression}, d1={d1}, d2={d2}\n')
    #                     f.write(f'\n{i} {extract_watermark(name, lenght_watermark=511, passwordwm=123456789, d1 = d1, d2 = d2)}, compression = {compression}, d1={d1}, d2={d2}\n')
    #                 except ValueError:
    #                     print(f"ValueError: non-hexadecimal number found in fromhex() arg at position 127 for compression = {compression}, d1={d1}, d2={d2}\n")
    #                     f.write(f"\nValueError: non-hexadecimal number found in fromhex() arg at position 127 for compression = {compression}, d1={d1}, d2={d2}\n")
    #                 finally:
    #                     continue
    # # toc = time.perf_counter()
    # # print(f"Извлечение multithreading заняло {toc - tic:0.1f} секунд") 