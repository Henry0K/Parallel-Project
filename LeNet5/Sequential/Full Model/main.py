import struct
import numpy as np
import sys

def read_idx(filename):
    """
    Reads an IDX file and returns the data as a NumPy array.
    """
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)
    return data

def pad_image(image, target_size=32):
    """
    Pads a 28x28 image to target_size x target_size (default 32x32) with zeros.
    Centers the original image.
    """
    padded_image = np.zeros((target_size, target_size), dtype=np.uint8)
    start = (target_size - image.shape[0]) // 2
    padded_image[start:start+image.shape[0], start:start+image.shape[1]] = image
    return padded_image

def save_binary_image(image, filename):
    """
    Saves a 32x32 image as a binary file with pixel values as bytes.
    """
    with open(filename, 'wb') as f:
        f.write(image.tobytes())

def main():
    if len(sys.argv) < 4:
        print("Usage: python extract_mnist_image.py <images_file> <labels_file> <index> [output_file]")
        print("Example: python extract_mnist_image.py train-images-idx3-ubyte train-labels-idx1-ubyte 0 output_image.bin")
        sys.exit(1)
    
    images_file = sys.argv[1]
    labels_file = sys.argv[2]
    index = int(sys.argv[3])
    output_file = sys.argv[4] if len(sys.argv) >=5 else f'image_{index}.bin'

    # Read images and labels
    images = read_idx(images_file)
    labels = read_idx(labels_file)

    if index < 0 or index >= len(labels):
        print(f"Index out of range. There are {len(labels)} images.")
        sys.exit(1)

    # Extract the image and label
    image = images[index]
    label = labels[index]
    print(f"Extracting image at index {index} with label {label}.")

    # Pad the image to 32x32
    padded_image = pad_image(image, target_size=32)

    # Save the image as a binary file
    save_binary_image(padded_image, "./infer/"+output_file)
    print(f"Saved padded image to {output_file}.")

    label_file = output_file.replace('.bin', '_label.txt')
    with open("./infer/"+label_file, 'w') as f:
        f.write(str(label))
    print(f"Saved label to {label_file}.")

if __name__ == "__main__":
    main()
