
# Python + PyCUDA implementation (prototype)
import numpy as np
import cv2
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule

kernel_src = r'''
__global__ void randomize_pixels(unsigned char *img, float *brightness, float *contrast, float *hue, int numPixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPixels) return;
    int off = idx*3;
    float r = img[off] / 255.0f;
    float g = img[off+1] / 255.0f;
    float b = img[off+2] / 255.0f;
    float c = contrast[idx];
    float br = brightness[idx];
    r = (r - 0.5f) * c + 0.5f + br;
    g = (g - 0.5f) * c + 0.5f + br;
    b = (b - 0.5f) * c + 0.5f + br;
    r = fminf(fmaxf(r,0.0f),1.0f);
    g = fminf(fmaxf(g,0.0f),1.0f);
    b = fminf(fmaxf(b,0.0f),1.0f);
    img[off] = (unsigned char)(r * 255.0f);
    img[off+1] = (unsigned char)(g * 255.0f);
    img[off+2] = (unsigned char)(b * 255.0f);
}
'''

mod = SourceModule(kernel_src)
func = mod.get_function('randomize_pixels')

def run(input_path, num_copies=1000):
    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    numPixels = h*w
    flat = img.reshape(-1).astype(np.uint8)
    # allocate device buffers
    img_gpu = drv.mem_alloc(flat.nbytes)
    br = np.zeros(numPixels, dtype=np.float32)
    ct = np.ones(numPixels, dtype=np.float32)
    hu = np.zeros(numPixels, dtype=np.float32)
    br_gpu = drv.mem_alloc(br.nbytes); ct_gpu = drv.mem_alloc(ct.nbytes); hu_gpu = drv.mem_alloc(hu.nbytes)
    threads = 256
    blocks = (numPixels + threads - 1) // threads
    for i in range(num_copies):
        # sample params
        bval = np.float32(np.random.uniform(-0.15, 0.15))
        cval = np.float32(np.random.uniform(0.6, 1.4))
        hval = np.float32(np.random.uniform(-0.4, 0.4))
        br.fill(bval); ct.fill(cval); hu.fill(hval)
        drv.memcpy_htod(img_gpu, flat)
        drv.memcpy_htod(br_gpu, br); drv.memcpy_htod(ct_gpu, ct); drv.memcpy_htod(hu_gpu, hu)
        func(img_gpu, br_gpu, ct_gpu, hu_gpu, np.int32(numPixels), block=(threads,1,1), grid=(blocks,1))
        drv.memcpy_dtoh(flat, img_gpu)
    out = flat.reshape((h,w,3))[:, :, ::-1]
    cv2.imwrite('randomized_pycuda.png', out)

if __name__ == '__main__':
    import sys
    run(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 1000)
