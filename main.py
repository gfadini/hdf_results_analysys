#!/usr/bin/python
import h5py
import matplotlib
import crocoddyl
import example_robot_data
import pinocchio
import sys
import time
import os
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np

def readFromPath(filePath):
    results_archive = h5py.File(filePath, mode='r')
    return results_archive

def cleanName(filePath):
    prettyName = filePath
    for uglyStuff in ['results/', '.npz', '.hdf5']:
        prettyName = prettyName.replace(uglyStuff, '')
    return prettyName

def getBest(archive):
    return archive

def extract(archive, tag, index=0):
    '''
    Function used to handle the multidimensionality of the search space
    It extracts a specific component of the saved data
    it handles the exception in which the index is an integer
    '''
    tmp_array = []
    for i in archive[tag]:
        try:
            tmp_array.append(i[index])
        except:
            tmp_array.append(i)
    return np.array(tmp_array)

def simplePlot(x, y, image_folder, extension, figTitle='plot', color='blue', plotTitle = '', xlabel='x', ylabel='y', quiet = True):
    '''
    Generic scatter or line plot
    '''
    plt.figure(figTitle)
    if y is not None and x is not None:
        plt.scatter(x, y, color = color, s=2**2)
        plt.ylim(bottom=min(y))
    elif y is None and x is not None :
        plt.plot(x, linestyle='none', marker = 'o', markersize = 2, color=color)
    else:
        raise Exception('Data is needed for the plot!')
    plt.title(plotTitle)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if image_folder is not None:
        plt.savefig(image_folder + figTitle + '.' + extension, format = extension)
    if not quiet:
        plt.show()

def cmapPlot(x, y, z, image_folder, extension, figTitle='scatter', cmap='BuPu_r', plotTitle = 'scatter', xlabel='x', ylabel='y', quiet = True):
    '''
    Default 2D colormap plot
    '''
    plt.figure(figTitle)
    scatter = plt.scatter(x, y, c = z, cmap = 'BuPu_r', s=3**2)
    plt.colorbar(scatter)
    plt.title(plotTitle)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(bottom=np.min(y))
    if image_folder is not None:
        plt.savefig(image_folder + figTitle + '.' + extension, format = extension)
    if not quiet:
        plt.show()

def plot_codesign_results(archive, image_folder = None, extension = 'pdf'):

    '''
    Creates the saving directory if needed
    '''
    if image_folder is not None:
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)

    cost = np.log(extract(archive, 'cost'))
    error =  np.log(extract(archive, 'error'))

    for index in range(archive['motor_mass'].shape[1]):
        motorMass = extract(archive, 'motor_mass', index)
        gearRatio = extract(archive, 'n_gear', index)
        scaling = extract(archive, 'lambda_l', index)

        label = str(index)

        simplePlot(motorMass, cost, image_folder, extension, 'cost_mass_' + label, 'blue', 'Cost and motor mass ' + label, '$m_m \; [$Kg$]$', '$\log(cost)$')
        simplePlot(gearRatio, cost, image_folder, extension, 'cost_trasmission_' + label, 'red', 'Cost and transmission ' + label, '$n \; [\;]$', '$\log(cost)$')
        simplePlot(scaling, cost, image_folder, extension, 'cost_scale_' + label, 'orange', 'Cost and scaling ' + label, '$\lambda_l \; [\;]$', '$\log(cost)$')

    simplePlot(cost, None, image_folder, extension, 'cost_evo', 'blue', 'Cost evolution during the optimization', 'Number of Iteration', '$\log(cost)$')
    simplePlot(error, None, image_folder, extension, 'error_evo', 'red', 'Error evolution during the optimization', 'Number of Iteration', '$\log(error)$')

    # cmapPlot(scaling, motorMass, cost, image_folder, extension, 'motor_lambda', 'BuPu_r', 'Motor mass and scaling', '$\\lambda_l$ [ ]', '$m_m$ [Kg]')
    # cmapPlot(motorMass, gearRatio, cost, image_folder, extension, 'motor_transmission', 'BuPu_r', 'Motor mass and gear ratio', '$m_m$ [Kg]', '$n$ [ ]')
    # cmapPlot(gearRatio, scaling, cost, image_folder, extension, 'transmission_lambda', 'BuPu_r', 'Gear ratio and scaling', '$n$ [ ]', '$\\lambda_l$ [ ]')

def selectSolution(archive, key, index):
    return np.array(archive[key][index, :])

def returnChampionIndex(archive):
    return archive['cost']

def generationMeans(archive, tag, gen = 100):
    means=[]
    N = len(archive['cost'])
    for i in range(0,int(N/gen)):
        means.append(np.mean(archive['cost'][i*int(gen):(i+1)*int(gen)]))
    return means

if __name__ == "__main__":
    files = sys.argv[1:]
    for filePath in files:
        #filePath = str(sys.argv[1])
        print('#'*20)
        prettyName = cleanName(filePath)
        print(prettyName)
        results_archive = readFromPath(filePath)

        plt.ioff()
        plot_codesign_results(results_archive, 'plots/' + prettyName + '/', extension = 'png')
        plt.close('all')
        solved = np.array(results_archive['solved'])
        print('Solved {:.2f}% of the problems [{:0}]'.format(sum(solved==1)/len(solved)*1e2, len(solved)))
        minVect = np.array(results_archive['champion_x'])
        minValue = np.array(results_archive['champion_f'])
        print('Minimum {:} found at {:}'.format(minValue, minVect))
        # champion.solve()
        # plot_power(champion)
        # plot_solution(champion)
        # plot_frame_trajectory(champion, 'tip')


