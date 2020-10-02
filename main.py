#!/usr/bin/python
import pinocchio
import sys
import time
import os
import numpy as np
import pendulum, modify_model, initialize_problem
from archive_utils import *
import ddp_utils

if __name__ == "__main__":
    # can specify multiple files as arguments
    # ideally a batch with wildcards
    files = sys.argv[1:]
    for filePath in files:

        prettyName = cleanName(filePath, ['results/', '.npz', '.hdf5'])
        results_archive = readFromPath(filePath)

        robot_model = pendulum.createPendulum(nbJoint=results_archive['motor_mass'].shape[1])

        are_solved = np.array(results_archive['solved'])
        try:
            acceptable_tolerance = np.array(results_archive['th_stop']) < 1e-8 #results_archive['th_stop_tol']
        except:
            acceptable_tolerance = np.array(results_archive['th_stop']) < 1e-9
        acceptable_cost = np.array(results_archive['cost']) < np.mean(np.array(results_archive['cost']))
        acceptable_error = np.array(results_archive['error']) < 1e-3
        acceptable_solutions = are_solved # acceptable_error * acceptable_tolerance * acceptable_cost * are_solved
        print('The acceptable solutions are {:2.2f}%'.format(100 * sum(acceptable_solutions)/results_archive['cost'].shape[0]))

        if sum(acceptable_solutions) == 0:
            best_acceptable = getChampionIndex(results_archive)
        else:
            cost_acceptable = results_archive['cost'][acceptable_solutions]
            error_acceptable = results_archive['error'][acceptable_solutions]
            best_acceptable = np.where(np.array(results_archive['cost']) == np.amin(cost_acceptable))[0][0]

        print('#'*20)
        print(prettyName)
        print('Solved {:.2f}% of the problems [{:0}]'.format(sum(are_solved==1)/len(are_solved)*1e2, len(are_solved)))
        print('Convergence evolution \n {:}'.format(generationMeans(results_archive, 'cost', 6e3)))
        print('Minimum {:} found at \n m_m {:} \n n {:}\n l_l {:}'.format(
                                                results_archive['cost'][best_acceptable],
                                                results_archive['motor_mass'][best_acceptable],
                                                results_archive['n_gear'][best_acceptable],
                                                results_archive['lambda_l'][best_acceptable]))

        image_folder = 'plots/' + prettyName + '/'

        plot_codesign_results(results_archive, acceptable_solutions, None, None, quiet = False)
        plotPower(results_archive, best_acceptable, None, None, quiet = False)
        plotSolution(results_archive, best_acceptable, None, None, quiet = False)
        frames=[frame.name for frame in robot_model.frames]
        animateSolution(results_archive, best_acceptable, robot_model.copy(), frameNames=frames, target=np.array([0,0,1]), dt = 1e-2)
        plot_frame_trajectory(results_archive, best_acceptable, robot_model.copy(), frame_names=frames, target=np.array([0,0,1]), trid = False, quiet = False)
        energy_stats(results_archive, best_acceptable, robot_model.copy(), dt=1e-2)

        plt.figure('histogram')
        _ = plt.hist(np.log(np.array(results_archive['cost'])[acceptable_solutions]), bins='auto')
        plt.show()

        # plotTrajectories(results_archive, frames=['tip'], selectedIndexes=(acceptable_solutions*range(len(acceptable_solutions))), quiet=False)
        plt.close('all')

        class configurationParams():
            def __init__(self):
                self.dt = 2e-3
                self.T = int(1e3)
                self.target = [0, 0, 1]
                self.nbJoint = 1
                self.weight_gripperPose = 1e3
                self.weight_finalVelocity = self.weight_gripperPose
                self.weight_power_losses_cost_correction = 1e-1
                self.weight_friction_power_cost = self.weight_power_losses_cost_correction
                self.th_stop = 1e-6
                self.max_iter = int(1e3)

            def overwrite(self, archive):
                for attribute in self.__dict__.keys():
                    try:
                        data = np.array(archive[attribute])
                        entry_type = type(self.__getattribute__(attribute))
                        if  entry_type == int or entry_type == float:
                            data = np.asscalar(data)
                        self.__setattr__(attribute, data)
                    except:
                        print('Key {:} not found in the stored archive, to default'.format(attribute))

        conf = configurationParams()
        conf.overwrite(results_archive)
        ddp = solve_ddp(results_archive, best_acceptable, conf)
        ddp_utils.plotOCSolution(ddp)
        ddp_utils.plotConvergence(ddp)
        ddp_utils.plot_power(ddp)
        ddp_utils.plot_frame_trajectory(ddp, 'tip')
