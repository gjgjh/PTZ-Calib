# PTZ-Calib

**PTZ-Calib** is a robust two-stage PTZ camera calibration method, that efficiently and accurately estimates camera parameters for arbitrary viewpoints.
Our method includes an offline and an online stage. In the offline stage, we first uniformly select a set of reference images that sufficiently overlap to encompass a complete 360◦
view. We then utilize the novel PTZ-IBA (PTZ Incremental Bundle Adjustment) algorithm to automatically calibrate the cameras within a local coordinate system. Additionally, for
practical application, we can further optimize camera parameters and align them with the geographic coordinate system using extra global reference 3D information. 
In the online stage, we formulate the calibration of any new viewpoints as a relocalization problem.

Please check [Our paper](https://arxiv.org/pdf/2502.09075) for more details.

Code and datasets will be released soon.
