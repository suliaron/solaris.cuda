solaris.cuda
============

NBody code for cuda

Authors: Áron Süli and László Dobos

Department of Astronomy and Complex Physics, Eötvös University, Budapest, Hungary

gSOLARIS is a successor of SOLARIS implemented to utilize cutting edge technology based on the graphical 
processing unit (GPU). Over the past decades, the $N^2$ nature of the direct force calculation has proved 
a barrier for extending the particle number. After an era of GRAPE computers which speeded up the force 
calculation further, we are now in the age of GPUs where relatively small hardware systems are highly 
cost effective. All operations are performed in parallel like the different force computations as well
as the integration step. It is a general purpose software package designed to integrate planet and
planetesimal dynamics in the early/late stage of planet formation. gSOLARIS is capable to (i) to follow
the orbital evolution of the solar system's major planets and minor bodies, (ii) to study the dynamics
of exoplanetary systems, and (iii) to study the early and later phases of planetary formation. Apart
from the Newtonian gravitational forces, aerodynamic drag force, and type I and II migration forces
are also implemented. The code also includes a nebula model. To further accelerate the computation,
gSOLARIS defines particles with different interaction properties. Several two–body events are monitored,
such as collision, ejection etc. gSOLARIS is written in C++/CUDA C and and runs on all Nvidia GPUs with
compute capability of at least 2.0. gSOLARIS is designed to be versatile and easy to use software package
for the scientific community.

This work was supported by the Hungarian Fund for Scientific Research, Grant no. OTKA-103244
