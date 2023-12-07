import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
def gaussian_filter(img, K_size=5, sigma=25.0):
    img = np.asarray(np.uint8(img))
    if len(img.shape) == 3:
        H, W, C = img.shape
    else:
        img = np.expand_dims(img, axis=-1)
        H, W, C = img.shape
 
    ## Zero padding
    pad = K_size // 2
    out = np.zeros((H + pad * 2, W + pad * 2, C), dtype=np.float64)
    out[pad: pad + H, pad: pad + W] = img.copy().astype(np.float64)
 
    ## prepare Kernel
    K = np.zeros((K_size, K_size), dtype=np.float64)
    for x in range(-pad, -pad + K_size):
        for y in range(-pad, -pad + K_size):
            K[y + pad, x + pad] = np.exp( -(x ** 2 + y ** 2) / (2 * (sigma ** 2)))
    K /= (2 * np.pi * sigma * sigma) 
    K /= K.sum()
    tmp = out.copy()
 
    # filtering
    for y in range(H):
       for x in range(W):
            for c in range(C): 
                out[pad + y, pad + x, c] = np.sum(K * tmp[y: y + K_size, x: x + K_size, c])
    out = np.clip(out, 0, 255)
    out = out[pad: pad + H, pad: pad + W].astype(np.uint8)
    return out
def gauss(K_size,sigma):
    pad = K_size // 2
    K = np.zeros((K_size, K_size), dtype=np.float64)
    for x in range(-pad, -pad + K_size):
        for y in range(-pad, -pad + K_size):
            K[y + pad, x + pad] = np.exp( -(x ** 2 + y ** 2) / (2 * (sigma ** 2)))
    K /= (2 * np.pi * sigma * sigma) 
    K /= K.sum()
    return K
if __name__ == "__main__":
    # img=cv.imread(r"C:\Users\XiYi\Desktop\dog222.jpeg")
    # out=gaussian_filter(img)
    # cv.imwrite('sigma=25.jpg',out)
    # cv.imshow('result',out)
    # cv.waitKey()
    plt.subplot(1,3,1)
    plt.title('Size=25,Sigma=1')
    plt.imshow(gauss(25,1))
    plt.subplot(1,3,2)
    plt.title('Size=25,Sigma=5')
    plt.imshow(gauss(25,5))
    plt.subplot(1,3,3)
    plt.title('Size=25,Sigma=7')
    plt.imshow(gauss(25,7))
    plt.tight_layout()
    plt.show()