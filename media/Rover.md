# Autonomous Rover

## Equations of Motion

We work the the problem of estimating the location of an autonomous rover with non-linear dynamics given by:

$$\begin{bmatrix}
x_k \\
y _k\\
\theta_k 
\end{bmatrix}=\begin{bmatrix}
x_{k-1}+T_s v_{k-1}\cos(\theta_{k-1}) \\
y_{k-1}+T_s v_{k-1}\sin(\theta_{k-1}) \\
\theta_{k-1} + T_s v_{k-1}\frac{\phi_{k-1}}{d} \\
\end{bmatrix}+\mathbf{w}_{k}$$

where $d=0.5$ m is the length of the robot, and $\phi_{k-1}$ is the steering angle. The process noise $\mathbf{w}_{k}$ is zero-mean Gaussian with covariance $Q=\text{diag}([0.2^2, 0.2^2, 0.05^2])$ and the sampling frequency is 2 Hz.

## Measurement Model

To aid its localisation, the vehicle also receives distance and bearing measurements from a stationary beacon at the coordinates $(x_B,y_B)$ = (20, 15). 

$$
\begin{bmatrix}
r_{k} \\
\beta_{k}
\end{bmatrix}=\begin{bmatrix}
\sqrt{(x_k-x_B)^2+(y_k-y_B)^2} \\
\text{atan2}\left(y_k-y_B,x_k-x_B\right)-\theta_k
\end{bmatrix}
$$

where the measurement noise $\mathbf{w}_{k}$ is zero-mean Gaussian with covariance $R=\text{diag}([1^2, 0.04^2])$.

We already implement an "RoverRobot" class that contains the dynamic model of the vehicle (f), the measurement model (h), and the Jacobian of the motion (F_k) and measurement (H_k) models. 

