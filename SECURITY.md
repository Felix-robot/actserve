# Security and robot-safety policy

ActServe v0.1 schedules inference requests. It does not provide motor
drivers, collision checking, watchdogs, emergency stops, or a complete robot
safety system.

Never connect model outputs directly to physical actuators without an
independent, hardware-specific safety controller and authorized on-site owner.

Please report software vulnerabilities privately to the repository maintainer.
Do not include credentials, private observations, model weights, or physical
robot access details in a public issue.
