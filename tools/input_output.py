import struct
import numpy as np

def float_to_bytes(f: float):
    return struct.pack('<d', f)

def int_to_bytes(i: int):
    return int.to_bytes(i, 8, 'little')

def float_from_bytes(f: float):
    return struct.unpack('<d', f)

def int_from_bytes(i: int):
    return int.from_bytes(i, 'little')

def load(path):
    with open(path, 'rb') as file:
        d = int_from_bytes(file.read(8))
        n = int_from_bytes(file.read(8))
        
        X = []

        for i in range(n): 
            arr = []
            for j in range(d):
                arr.append(float_from_bytes(file.read(8)))
            X.append(arr)

    return np.array(X).reshape(n,d)
                
def save(X, path):
    d = len(X[0])
    n = len(X)

    out_data = int_to_bytes(d) + int_to_bytes(n)

    for x in X: 
        for c in x:
            out_data += float_to_bytes(c)

    with open(path, 'wb') as file:
        file.write(out_data)