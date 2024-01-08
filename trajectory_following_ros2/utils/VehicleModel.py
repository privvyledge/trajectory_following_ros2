import matplotlib
matplotlib.use('TkAgg')
import math
import numpy as np


class VehicleState:
    def __init__(self, x=0.0, y=0.0, z=0.0, yaw=0.0,
                 vel=0.0, yaw_rate=0.0,
                 steering_angle=0.0, beta=0.0, wheelbase=0.256):
        """
        Define a vehicle state class
        :param x: float, x position
        :param y: float, y position
        :param yaw: float, vehicle heading
        :param vel: float, velocity
        """
        self.x = x
        self.y = y
        self.z = z
        self.yaw = yaw
        self.delta = steering_angle
        self.vel = vel
        self.yaw_rate = yaw_rate
        self.beta = beta
        self.rear_x = self.x - ((wheelbase / 2) * math.cos(self.yaw))
        self.rear_y = self.y - ((wheelbase / 2) * math.sin(self.yaw))

    def update_states(self, acc, delta, wheelbase, dt=0.01):
        """
        Vehicle motion model (in the global frame), here we are using simple bicycle model
        :param acc: float, acceleration
        :param delta: float, heading control
        """
        self.x += self.vel * math.cos(self.yaw) * dt
        self.y += self.vel * math.sin(self.yaw) * dt
        self.yaw += self.vel * math.tan(delta) / wheelbase * dt
        self.vel += acc * dt
        # if x, and y are relative to CoG
        self.rear_x = self.x - ((wheelbase / 2) * math.cos(self.yaw))
        self.rear_y = self.y - ((wheelbase / 2) * math.sin(self.yaw))


class Vehicle(object):
    """docstring for ClassName"""

    def __init__(self, mass=0.0, yaw_moment_of_inertia=0.0,
                 wheelbase=0.256, length=4.5, width=2.0, front_track=0.0, rear_track=3.3,
                 max_steering=45.0, min_steering=-45.0,
                 max_steering_rate=30.0, max_speed=55.0 / 3.6, min_speed=-20.0 / 3.6,
                 max_accel=1.0, max_decel=-1.0,
                 state=VehicleState(x=0.0, y=0.0, z=0.0, yaw=0.0, vel=0.0, steering_angle=0.0, yaw_rate=0.0, beta=0.0, wheelbase=0.256),
                 n_states=4, n_inputs=2):
        """Constructor for Vehicle"""
        # constant vehicle parameters.
        self.MASS = mass
        self.YAW_INERTIA_MOMENT = yaw_moment_of_inertia

        #
        self.WHEELBASE = wheelbase  # [m]
        self.LENGTH = length  # [m]
        self.WIDTH = width  # [m] width of vehicle
        self.WD = 0.7 * self.WIDTH  # [m] distance between left-right wheels
        self.RF = front_track  # [m] distance from rear to vehicle front end of vehicle
        self.RB = rear_track  # [m] distance from rear to vehicle back end of vehicle
        # self.BACKTOWHEEL = BACKTOWHEEL  # [m]
        # self.WHEEL_LEN = WHEEL_LEN  # [m]
        # self.WHEEL_WIDTH = WHEEL_WIDTH  # [m]
        # self.TREAD = TREAD  # [m]
        # self.TR = TR  # [m] Tyre radius
        # self.TW = TW  # [m] Tyre width

        # Limits
        self.MAX_STEER = np.deg2rad(max_steering)  # maximum steering angle [rad]
        self.MIN_STEER = np.deg2rad(min_steering)  # minimum steering angle [rad]. Could be the same as max
        self.MAX_DSTEER = np.deg2rad(max_steering_rate)  # maximum steering speed [rad/s]
        self.MAX_SPEED = max_speed  # maximum speed [m/s]
        self.MIN_SPEED = min_speed  # minimum speed [m/s]
        self.MAX_ACCEL = max_accel  # maximum accel [m/ss]
        self.MAX_DECEL = max_decel  # maximum deceleration [m/ss]

        # State/position
        self.state = state
        self.NX = n_states  # state vector: z (or x) = [x, y, v, phi/yaw]
        self.NU = n_inputs  # input vector: u (or a) = [accel, steer]

    def limit_velocity(self, desired_velocity):
        velocity = np.clip(desired_velocity, self.MIN_SPEED, self.MAX_SPEED)
        return velocity

    def limit_steering(self, desired_steering_angle):
        steering_angle = np.clip(desired_steering_angle, self.MIN_STEER, self.MAX_STEER)
        return steering_angle

    def update_state(self, acc, delta, dt=0.2, model='kinematic'):
        """
        Vehicle motion model (in the global frame), here we are using simple bicycle model
        :param acc: float, acceleration
        :param delta: float, heading control
        """
        if model == 'kinematic':
            self.update_states_kinematic(acc, delta, dt)

        else:
            self.update_states_dynamic(acc, delta, dt)

    def update_states_kinematic(self, acc, delta, dt=0.2):
        delta = self.limit_steering(delta)
        self.state.x += self.state.vel * math.cos(self.state.yaw) * dt
        self.state.y += self.state.vel * math.sin(self.state.yaw) * dt
        self.state.yaw += self.state.vel * math.tan(delta) / self.WHEELBASE * dt
        self.state.vel += acc * dt
        self.state.vel = self.limit_velocity(self.state.vel)
        # if x, and y are relative to CoG
        self.state.rear_x = self.state.x - ((self.WHEELBASE / 2) * math.cos(self.state.yaw))
        self.state.rear_y = self.state.y - ((self.WHEELBASE / 2) * math.sin(self.state.yaw))

    def update_states_dynamic(self, acc, delta, dt=0.2):
        # see (https://github.com/f1tenth/f1tenth_planning/blob/main/f1tenth_planning/control/dynamic_mpc/dynamic_mpc.py#L317)
        # see (https://github.com/AtsushiSakai/PyAdvancedControl/blob/master/steer_vehicle_model/dynamic_bicyccle_model.py#L32)
        pass

    def get_model_matrix(self, v, phi, delta, dt=0.2, model='kinematic'):
        if model == 'kinematic':
            A, B, G = self.get_linear_model_matrix(v, phi, delta, dt=dt)
        else:
            A, B, G = self.get_dynamic_model_matrix(v, phi, delta, dt=dt)

        return A, B, G

    def get_linear_model_matrix(self, v, phi, delta, dt=0.2):
        """
        calc linear and discrete time dynamic model -> Explicit discrete time-invariant
        Linear System: Xdot = Ax + Bu + C
        State vector: x=[x, y, v, yaw]

        :param v: speed: v_bar
        :param phi: heading angle of vehicle: phi_bar
        :param delta: steering angle: delta_bar
        :return: A, B, C
        """
        # State (or system) matrix A, 4x4
        A = np.zeros((self.NX, self.NX))
        A[0, 0] = 1.0
        A[1, 1] = 1.0
        A[2, 2] = 1.0
        A[3, 3] = 1.0
        A[0, 2] = dt * math.cos(phi)
        A[0, 3] = - dt * v * math.sin(phi)
        A[1, 2] = dt * math.sin(phi)
        A[1, 3] = dt * v * math.cos(phi)
        A[3, 2] = dt * math.tan(delta) / self.WHEELBASE

        # A = np.array([[1.0, 0.0, dt * math.cos(phi), - dt * v * math.sin(phi)],
        #                [0.0, 1.0, dt * math.sin(phi), dt * v * math.cos(phi)],
        #                [0.0, 0.0, 1.0, 0.0],
        #                [0.0, 0.0, dt * math.tan(delta) / self.WHEELBASE, 1.0]])

        # Input Matrix B; 4x2
        B = np.zeros((self.NX, self.NU))
        B[2, 0] = dt
        B[3, 1] = dt * v / (self.WHEELBASE * math.cos(delta) ** 2)

        # B = np.array([[0.0, 0.0],
        #                [0.0, 0.0],
        #                [dt, 0.0],
        #                [0.0, dt * v / (self.WHEELBASE * math.cos(delta) ** 2)]])

        C = np.zeros(self.NX)
        C[0] = dt * v * math.sin(phi) * phi
        C[1] = - dt * v * math.cos(phi) * phi
        C[3] = - dt * v * delta / (self.WHEELBASE * math.cos(delta) ** 2)

        # C = np.array([dt * v * math.sin(phi) * phi,
        #                -dt * v * math.cos(phi) * phi,
        #                0.0,
        #                -dt * v * delta / (self.WHEELBASE * math.cos(delta) ** 2)])

        return A, B, C

    def get_dynamic_model_matrix(self, v, phi, delta, dt=0.2):
        # See https://github.com/f1tenth/f1tenth_planning/blob/main/f1tenth_planning/control/dynamic_mpc/dynamic_mpc.py#L428
        return None, None, None

    def predict_motion(self, x0, oa, od, xref, horizon, model='kinematic'):
        if model == 'kinematic':
            xbar = self.predict_motion_kinematic(x0, oa, od, xref, horizon)
        else:
            xbar = self.predict_motion_dynamic(x0, oa, od, xref, horizon)

        return xbar

    def predict_motion_kinematic(self, x0, oa, od, xref, horizon):
        """
        Todo: move to Vehicle Class
        given the current state, using the acceleration and delta strategy of last time,
        predict the states of vehicle in T steps.
        :param x0 (z0): initial state
        :param oa: acceleration strategy of last time
        :param od (delta): delta strategy of last time
        :param xref (z_ref): reference trajectory
        :return: predict states in T steps (z_bar, used for calc linear motion model)
        """
        xbar = xref * 0.0
        for i, _ in enumerate(x0):
            xbar[i, 0] = x0[i]

        """
        self.model.state.x = x0[0]
        self.model.state.y = x0[1]
        self.model.state.yaw = x0[3]
        self.model.state.vel = x0[2]
        for (ai, di, i) in zip(oa, od, range(1, T + 1)):
            # state = update_state(state, ai, di)
            self.model.update_state(ai, di, dt=0.2)
            xbar[0, i] = self.model.state.x
            xbar[1, i] = self.model.state.y
            xbar[2, i] = self.model.state.vel
            xbar[3, i] = self.model.state.yaw
        """

        state = VehicleState(x=x0[0], y=x0[1], yaw=x0[3], vel=x0[2],
                             steering_angle=0.0, yaw_rate=0.0, beta=0.0, wheelbase=self.WHEELBASE)
        model = Vehicle(wheelbase=self.WHEELBASE, max_steering=self.MAX_STEER, min_steering=self.MIN_STEER,
                        max_steering_rate=self.MAX_DSTEER, max_speed=self.MAX_SPEED, min_speed=self.MIN_SPEED,
                        state=state)
        for (ai, di, i) in zip(oa, od, range(1, horizon + 1)):
            # state = update_state(state, ai, di)
            model.update_state(ai, di, dt=0.2)
            xbar[0, i] = model.state.x
            xbar[1, i] = model.state.y
            xbar[2, i] = model.state.vel
            xbar[3, i] = model.state.yaw

        return xbar

    def predict_motion_dynamic(self, x0, oa, od, xref, horizon):
        pass
