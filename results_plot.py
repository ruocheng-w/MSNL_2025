import numpy as np
import matplotlib.pyplot as plt  # plotting
from skimage import io, color, transform  # image processing: reading, grayscale, resizing
from sklearn.preprocessing import normalize  # normalises rows of a matrix
import pywt  # PyWavelets
import cvxpy as cp  # convex optimisation
from datetime import datetime
import time


# --- Step 1: Load and preprocess image ---
def load_image(path, size=(64, 64)):
    img = io.imread(path)  # loads image
    # If it has 4 channels, drop the alpha (transparency)
    if img.shape[-1] == 4:
        img = img[:, :, :3]
    img_gray = color.rgb2gray(img)  # convert to grayscale
    img_resized = transform.resize(img_gray, size, anti_aliasing=True)
    # resizes image with smooth transitions between pixels
    return img_resized


# --- Step 2: Sparsify with Wavelet Transform ---
def wavelet_sparsify(img):
    coeffs = pywt.wavedec2(img, 'haar', level=2)  # 2D DWT
    coeff_arr, coeff_slices = pywt.coeffs_to_array(coeffs)  # stores coeff in an array, remembers pos
    return coeff_arr, coeff_slices


def wavelet_unsparsify(coeff_arr, coeff_slices):  # inverse transform
    coeffs = pywt.array_to_coeffs(coeff_arr, coeff_slices, output_format='wavedec2')
    return pywt.waverec2(coeffs, 'haar')

# --- Step 2': Construct image array without Wavelet Transform
def vectorise(img):
    return img.T.flatten()

def unvectorise(vec, m):
    n = vec.shape[0] // m  # number of columns
    img = np.empty((m, n))
    for i in range(n):
        col = vec[i * m:i * m + m]
        img[:, i] = col
    return img



# --- Step 3: Create measurement matrix Phi and simulate y = Phi x ---
def compress_measurements(x, m):
    n = len(x)
    np.random.seed(42)
    Phi = np.random.randn(m, n)  # random Gaussian
    Phi = normalize(Phi, axis=1)  # make rows unit norm
    y = Phi @ x
    return y, Phi


# --- Step 4: L1 minimization to reconstruct sparse x ---
def reconstruct_l1(Phi, y):
    n = Phi.shape[1]  # number of columns, i.e. size of image vector
    s_hat = cp.Variable(n)
    objective = cp.Minimize(cp.norm1(s_hat))
    constraints = [Phi @ s_hat == y]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS)  
    return s_hat.value

# size m*n, c*m*n measurements
def compressed_sensing(M, N, c):
    start_time = time.time()
    img = load_image("your_file", size=(M, N))
    coeff_arr, coeff_slices = wavelet_sparsify(img)  # wavelet transform
    x = coeff_arr.flatten()
    # x = vectorise(img)  # without wavelet transform

    # Compressed sensing: take 30% measurements
    m = int(c * len(x))
    y, Phi = compress_measurements(x, m)

    # Reconstruct
    x_reconstructed = reconstruct_l1(Phi, y)
    coeffs_reconstructed = x_reconstructed.reshape(coeff_arr.shape)  # with wavelet transform
    img_reconstructed = wavelet_unsparsify(coeffs_reconstructed, coeff_slices)
    # img_reconstructed = unvectorise(x_reconstructed, N)

    # Show results
    MSE = 1 / (M * N) * np.linalg.norm(img - img_reconstructed, ord='fro')
    # fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    # print(MSE)
    # axs[0].imshow(img, cmap='gray')
    # axs[0].set_title("Original Image")
    # axs[1].imshow(np.clip(img_reconstructed, 0, 1), cmap='gray')
    # axs[1].set_title("Reconstructed Image " + str(c * 100) + "%")
    # for ax in axs: ax.axis('off')
    # plt.tight_layout()
    # fig.subplots_adjust(top=0.9)
    # filename = "your_file"
    # fig.savefig(filename)
    # plt.show()
    # plt.close()
    end_time = time.time()
    t = end_time - start_time
    print(f"Execution time: {t:.2f} seconds")
    return MSE

if __name__ == "__main__":
    k = 6
    M = 64
    N = 64
    c = 0.1
    X = np.arange(c, c * k, c)
    Y = []
    for i in range(k):
        Y.append(compressed_sensing(M, N, c * (i + 1)))
    Y = np.array(Y)
    plt.plot(X, Y, marker='o')
    plt.xlabel('Percentage')
    plt.ylabel('MSE')
    plt.title('MSE vs Percentage')
    plt.grid(True)
    name = "your_filef"
    plt.savefig(name)
    plt.show()
    plt.close()
