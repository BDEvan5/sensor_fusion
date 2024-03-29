# Sensor Fusion Algorithms

This repo provides examples of popular filtering algorithms for state estimation.
Currently, the following algorithms are availble:
- Extended Kalman Filter (EKF)
- Unscented Kalman Filter (UKF)
- Particle Filter (PF)

An in-depth note on probability theory and the state estimation equations is [provided](media/KalmanFilterNotes.pdf).

> 📘 Usage
> 
> Run the `RoverTests.py` and `RacingTests.py` files to test the filters.
> Run the `linear_fusion_tests.py` and `non_linear_fusion.py` to test the sensor fusion examples.

## Rover Tests

The task of localising a rover vehicle is used to evaluate the filtering algorithms.
The vehicle travels at constant speed with a steering angle input. 
Non-linear dynamics are used to represent the position updates, based on the rover orientation.
The rover receives measurement updates from a beacon that provides a distance and angle from the vehcile.
The task is to use these two sources of information to estimate the rovers position.
Full equations for the rover, and the derivation of the Jacobians are given in the [Rover](media/Rover.md) note.

## Racing Tests

Since the most common application of the particle filter is robot localisation, an example of autonomous racing is included.
The same rover dynamics are used, but a controller that moves the vehicle around a track is introduced.
Instead of a beacon, a LiDAR scanner is used and a method is provided to simulate LiDAR scans.
The goal of the task is to estimate the vehicle's position as accurately as possible.

**Particle Filter:**

![](media/ParticleFilter.gif)

**Extended Kalman Filter Sensor Fusion:**

![](media/SensorFusion_2D.gif)


## Results

Results from the rover localisation tests.

![](media/Extended%20Kalman%20Filter.svg)

![](media/Unscented%20Kalman%20Filter.svg)

![](media/Particle%20Filter.svg)

Particle filter, LiDAR based localisation:

![](media/Particle%20Filter%20localisation.svg)

### Sensor Fusion

Linear Sensor Fusion

![](media/LinearSensorFusion_comparison.svg)

Non-linear Sensor Fusion
![](media/NonLinearFusion_comparison.svg)

# Installation

We propose using a venv for package mangaement.




