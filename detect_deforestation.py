#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
import sys, os
from math import sqrt
import numpy as np
import argparse

def convertList(l):
    # Function to convert list items to integers
    return [int(item) for item in l]

def createDataset(path, c, dataset):
    # Extracts color information from images and generates a dataset.
    # Saves the names of the images in the passed directory to a file
    os.system(f'ls {path}/*ppm > filenames.txt')
    with open('filenames.txt', 'r') as f:
        names = f.readlines()

    # Removes formatting characters
    names = [name.strip('\n') for name in names]

    # For each training file
    for name in names:
        item = []
        with open(name, 'r') as img:
            img_data = img.read().replace('\n', ' ').split()
            red = convertList(img_data[4::3])
            green = convertList(img_data[5::3])
            blue = convertList(img_data[6::3])

            # Calculates the RGB averages of the entire image (8x8 pixels)
            avg = (sum(red)/len(red), sum(green)/len(green), sum(blue)/len(blue))

            # Saves the RGB values in a list, along with its classification
            item.extend([avg[0], avg[1], avg[2], c])
            dataset.append(item)

    # Removes the auxiliary file
    os.system('rm filenames.txt')

def readImage(image):
    # Reads the ppm image and organizes it into a suitable format
    # to save into a matrix
    with open(image, 'r') as img:
        img_data = img.read().replace('\n', ' ').split()

        # Each 'chunk' or line belonging to the 8x8 block, is composed
        # of 24 items (8 RGB triples), forming a block corresponding
        # to 64 pixels at the end
        chunk_size = int(img_data[1]) * 3
        aux = img_data[4:]
        header = img_data[:4]
        img_matrix = [aux[i:i + chunk_size] for i in range(0, len(aux), chunk_size)]

    return img_matrix, img_data, header

def everyPatch(image, block_size):
    # Divides the image into several 8x8 pixel patches
    width = len(image[0])  # Number of columns
    height = len(image)  # Number of rows
    position = (0, 0)  # Row x Column
    patches = []

    while True:
        patch = []

        for index in range(position[0], position[0] + block_size):
            patch.append(image[index][position[1]: position[1] + block_size * 3])

        patches.append(patch)
        if (position[1] + block_size * 3) < width:
            position = (position[0], position[1] + block_size * 3)
        else:
            if (position[0] + block_size) >= height:
                break
            else:
                position = (position[0] + block_size, 0)

    return patches

def euclideanDistance(x, y):
    # Calculates the Euclidean distance between elements
    return sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2 + (x[2] - y[2]) ** 2)

def classify(patches, k, block_size):
    global dataset
    for p in range(len(patches)):
        distance = []
        l = []
        for item in patches[p]:
            l += item

        red = convertList(l[0::3])
        green = convertList(l[1::3])
        blue = convertList(l[2::3])

        avg = (sum(red)/len(red), sum(green)/len(green), sum(blue)/len(blue))
        for item in dataset:
            distance.append((euclideanDistance(avg, item[:3]), item[3]))
        distance.sort(key=lambda d: d[0])
        d, f = 0, 0
        k = k - 1 if k % 2 == 0 else k
        k = max(1, k)

        for i in range(k):
            if distance[i][1] == 'd':
                d += 1
            else:
                f += 1
        if d > f:
            # Create an entirely red block to replace blocks classified as deforested
            patches[p] = [[255 if x % 3 == 0 else 0 for x in range(block_size * 3)] for y in range(block_size)]

    return patches

def rebuildImage(image, header, block_size):
    # Function that rebuilds the image with the classified blocks
    width = int(header[1])
    height = int(header[2])
    start = 0
    end = width // block_size
    lines = []

    for _ in range(height // block_size):
        # Creates a list with the lines of blocks that make up the image
        lines.append(np.hstack(image[start:end]))
        start += width // block_size
        end += width // block_size

    # Joins the block lines vertically, forming the image matrix
    rebuilt = np.vstack(lines)

    with open('output.ppm', 'w') as img:
        # Opens the output file and writes the ppm header
        img.write(f'{header[0]}\n')
        img.write(f'{header[1]} {header[2]}\n')
        img.write(f'{header[3]}\n')
        for j in rebuilt:
            for i in j:
                img.write(f'{i} ')

def main():
    parser = argparse.ArgumentParser(description='Image Classification')
    parser.add_argument('filename', type=str, help='Image file name')
    parser.add_argument('k', type=int, help='Value of K')
    parser.add_argument('block_size', type=int, help='Classification block dimensions* (size x size) \n* Both the height and width of the chosen image must be multiples of the block size passed as an argument, otherwise errors will occur.')

    args = parser.parse_args()

    global dataset
    dataset = []

    print("[+] Loading classification datasets...")

    # Create datasets for use in k-nn
    createDataset('deforestation', 'd', dataset)
    createDataset('forest', 'f', dataset)

    if args.k > len(dataset):
        print("The chosen value for k is too high! Increase the number of samples or decrease the value of k.")
        raise SystemExit

    print("[+] Performing classification...")
    # Image to be classified is passed by argument
    img_matrix, img_data, header = readImage(args.filename)
    # Get all 8x8 patches from the image
    patches = everyPatch(img_matrix, args.block_size)
    # Classify patches and return the highlighted image
    new_img = classify(patches, args.k, args.block_size)

    # Call the function that rebuilds the image
    rebuildImage(new_img, header, args.block_size)

    print("[+] Output saved as 'output.ppm'")

if __name__ == "__main__":
    main()
