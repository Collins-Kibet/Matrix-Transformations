'''scatter3d.py
3D scatterplot to help visualize letter dataset
Oliver W. Layton, Hannah Wolfe, Stephanie Taylor
CS 251 Data Analysis Visualization
Spring 2021

Running this file should generate a new interactive window with a 3D scatterplot of the letter data.
Clicking and dragging rotates the data.
'''
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import data


def scatter3d(data, headers=['X', 'Y', 'Z'], title='Raw letter data'):
    '''Creates a 3D scatter plot to visualize data'''
    letter_xyz = data.select_data(headers)

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    # Scatter plot of data in 3D
    ax.scatter3D(letter_xyz[:, 0], letter_xyz[:, 1], letter_xyz[:, 2])

    # style the plot
    ax.set_xlabel(headers[0])
    ax.set_ylabel(headers[1])
    ax.set_zlabel(headers[2])
    ax.set_title(title)

    plt.show()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python3 scatter3d.py <path to data CSV file>')
        print('Example: python3 scatter3d.py data/letter_data.csv')
        exit()
    letter_filename = 'data/letter_data.csv'
    letter_data = data.Data(letter_filename)
    scatter3d(letter_data)
