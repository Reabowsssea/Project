# HoloOcean

![HoloOcean Image](client/docs/images/inspect_plane.jpg)

[![pages-build-deployment](https://github.com/byu-holoocean/holoocean-docs/actions/workflows/pages/pages-build-deployment/badge.svg)](https://github.com/byu-holoocean/holoocean-docs/actions/workflows/pages/pages-build-deployment)
 [![Build Status](https://robots.et.byu.edu:4144/api/badges/byu-holoocean/HoloOcean/status.svg?ref=refs/heads/develop)](https://robots.et.byu.edu:4144/byu-holoocean/HoloOcean)



HoloOcean is a high-fidelity simulator develped by the [Field Robotic Systems Lab (FRostLab)](https://frostlab.byu.edu) at [Brigham Young University](https://byu.edu).

Built upon Unreal Engine (by Epic Games) and Holodeck (developed by the BYU PCCL Lab), HoloOcean enables easy simulation of marine robotics and autonomy with a wide variety of sensors, agents, and features.

## Features
 - 3+ rich worlds with various infrastructure for generating data or testing underwater algorithms
 - Complete with common underwater sensors including DVL, IMU, optical camera, various sonar, depth sensor, and more
 - Highly and easily configurable sensors and missions
 - Multi-agent missions, including optical and acoustic communications
 - Novel sonar simulation framework for simulating imaging, profiling, sidescan, and echosounder sonars
 - Imaging sonar implementation includes realistic noise modeling for small sim-2-real gap
 - Easy installation and simple, OpenAI Gym-like Python interface
 - High performance - simulation speeds of up to 2x real time are possible. Performance penalty only for what you need
 - Run headless or watch your agents learn
 - Linux and Windows support

## Installation
Clone the repo, then
`cd client`
`pip install .`

(requires >= Python 3.6)

See [Installation](https://byu-holoocean.github.io/holoocean-docs/usage/installation.html) for complete instructions (including Docker).

## Documentation
* [Quickstart](https://byu-holoocean.github.io/holoocean-docs/usage/getting-started.html)
* [Changelog](https://byu-holoocean.github.io/holoocean-docs/changelog/changelog.html)
* [Examples](https://byu-holoocean.github.io/holoocean-docs/usage/getting-started.html#code-examples)
* [Agents](https://byu-holoocean.github.io/holoocean-docs/agents/agents.html)
* [Sensors](https://byu-holoocean.github.io/holoocean-docs/holoocean/sensors.html)
* [Available Packages and Worlds](https://byu-holoocean.github.io/holoocean-docs/packages/packages.html)
* [Docs](https://byu-holoocean.github.io/holoocean-docs)

## Usage Overview
HoloOcean's interface is similar to OpenAI's gym.

We try and provide a batteries included approach to let you jump right into using HoloOcean, with minimal
fiddling required.

To demonstrate, here is a quick example using the `DefaultWorlds` package:

```python
import holoocean

# Load the environment. This environment contains a hovering AUV in a pier
env = holoocean.make("PierHarbor-Hovering")

# You must call `.reset()` on a newly created environment before ticking/stepping it
env.reset()                         

# The AUV takes commands for each thruster
command = [0, 0, 0, 0, 10, 10, 10, 10]   

for i in range(30):
    state = env.step(command)  
```

- `state`: dict of sensor name to the sensor's value (nparray).

If you want to access the data of a specific sensor, import sensors and
retrieve the correct value from the state dictionary:

```python
print(state["DVLSensor"])
```

## Attribution and Relevent Publications

In addition to the [online documentation for HoloOcean](https://byu-holoocean.github.io/holoocean-docs), 
the features developed in HoloOcean have also been disseminated via publication in peer-reviewed conferences and journal.
Please refer to these publications in addition to the documentation as needed.

If you use HoloOcean for your research, please cite the relevent publications depending on the features you use as outlined below:

### General HoloOcean Use
```
@inproceedings{Potokar22icra,
  author = {E. Potokar and S. Ashford and M. Kaess and J. Mangelson},
  title = {Holo{O}cean: An Underwater Robotics Simulator},
  booktitle = {Proc. IEEE Intl. Conf. on Robotics and Automation, ICRA},
  address = {Philadelphia, PA, USA},
  month = {May},
  year = {2022}
}
```

### Simulation of Sonar (Imaging, Sidescan, Profiling/Bathymetric)
```
@inproceedings{Potokar22iros,
  author = {E. Potokar and K. Lay and K. Norman and D. Benham and T. Neilsen and M. Kaess and J. Mangelson},
  title = {Holo{O}cean: Realistic Sonar Simulation},
  booktitle = {Proc. IEEE/RSJ Intl. Conf. Intelligent Robots and Systems, IROS},
  address = {Kyoto, Japan},
  month = {Oct},
  year = {2022}
}
```
