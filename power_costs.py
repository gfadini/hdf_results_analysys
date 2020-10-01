import crocoddyl
import numpy as np
import pinocchio
pinocchio.switchToNumpyArray()

class CostModelJointFriction(crocoddyl.CostModelAbstract):
    '''
        Describes the Coulomb friction power losses P_f
        T_f = T_mu sign(omega_m) + b omega_m ** 2 [Nm]
        P_f = T_f omega_m [W]
        the absolute value is approximated for better convergence
    '''
    def __init__(self, state, activation, nu):
        if not hasattr(state, 'robot_model'):
            raise Exception('State needs to have the model parameters, add the model to the state')
        self.T_mu = state.robot_model.T_mu
        self.b = state.robot_model.b
        self.n = state.robot_model.rotorGearRatio
        self.gamma = 1
        activation = activation if activation is not None else crocoddyl.ActivationModelQuad(state.ndx)
        crocoddyl.CostModelAbstract.__init__(self, state, activation, nu = nu)

    def calc(self, data, x, u):
        data.Tf = self.T_mu * np.tanh(self.gamma * x[self.state.nq:]) + self.b * x[self.state.nq:]
        data.cost = np.sum(data.Tf * x[self.state.nq:] * self.n)

    def calcDiff(self, data, x, u, recalc=True):
        if recalc:
            self.calc(data, x, u)
        data.dTfdx = self.T_mu * self.gamma * (1 - np.tanh(self.gamma*x[self.state.nq:])**2) + self.b
        data.d2Tfdx2 = 2 * self.T_mu * self.gamma**2 * (np.tanh(self.gamma*x[self.state.nq:])**2 - 1) * np.tanh(self.gamma*x[self.state.nq:])
        data.Lx[self.state.nv:] = self.n * (data.dTfdx * x[self.state.nq:] + data.Tf * np.ones(self.state.nv))
        data.Lxx[self.state.nv:, self.state.nv:] = np.diag(self.n * (data.d2Tfdx2 * x[self.state.nq:] + 2 * data.dTfdx))


class CostModelJouleDissipation(crocoddyl.CostModelAbstract):
    '''
        This cost is taking into account in the Joule dissipation P_t
        to the motor torque to drive to morion it's also added the Coulomb friction torque
        T_f = T_mu sign(omega_m) [Nm]
        P_t = (T_m + T_f).T [K] (T_m + T_f) [W]
    '''
    def __init__(self, state, activation, nu):
        if not hasattr(state, 'robot_model'):
            raise Exception('State needs to have the model parameters, add the model to the state')
        self.T_mu = state.robot_model.T_mu
        self.b = state.robot_model.b
        self.n = state.robot_model.rotorGearRatio
        self.K = np.array(1/state.robot_model.K_m)
        self.gamma = 1

        activation = activation if activation is not None else crocoddyl.ActivationModelQuad(state.ndx)
        crocoddyl.CostModelAbstract.__init__(self, state, activation, nu = nu)

    def calc(self, data, x, u):
        data.Tf = self.T_mu * np.tanh(self.gamma * x[self.state.nq:]) + self.b * x[self.state.nq:]
        data.Ttot = data.Tf + u / self.n
        data.cost = np.sum(self.K * data.Ttot**2)

    def calcDiff(self, data, x, u, recalc = True):
        if recalc:
            self.calc(data, x, u)
        # partial derivatives
        data.dTfdx = self.T_mu * self.gamma * (1 - np.tanh(self.gamma*x[self.state.nq:])**2) + self.b
        data.d2Tfdx2 = 2 * self.T_mu * self.gamma**2 * (np.tanh(self.gamma*x[self.state.nq:])**2 - 1) * np.tanh(self.gamma*x[self.state.nq:])
        data.Lx[self.state.nq:] = 2 * data.Ttot * self.K * data.dTfdx
        data.Lu[:] = 2 * self.K * data.Ttot / self.n

        data.Lxx[self.state.nq:, self.state.nq:] = np.diag(2 * (data.Ttot * self.K) * data.d2Tfdx2 + 2 * self.K * data.dTfdx**2)
        data.Luu[:,:] = np.diag(2 * self.K  / self.n**2)

        if self.nu > 1:
            data.Lxu[self.state.nq:, :] = 2 * np.diag(data.dTfdx * self.K / self.n)
        else:
            data.Lxu[self.state.nq:] = 2 * np.diag(data.dTfdx * self.K / self.n)
