# conda activate TesseracT

# pip install bitarray

import bitarray

ba = bitarray.bitarray()

text = 'Hoi'

ba.frombytes(text.encode('utf-8'))
print(ba)

# Li = ba.tolist()
# print(Li)

text_back = bitarray.bitarray(ba).tobytes().decode('utf-8')
print(text_back)