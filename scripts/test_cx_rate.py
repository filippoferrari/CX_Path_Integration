import numpy as np
import matplotlib.pyplot as plt

import central_complex
import cx_rate
import trials
import analysis
import plotter

T_outbound = 1500
T_inbound = 1500
noise = 0.1

h, v = trials.generate_route(T=T_outbound, vary_speed=True)
    
cx = cx_rate.CXRatePontin(noise=noise)

h, v, log, cpu4_snapshot = trials.run_trial(logging=True,
                                            T_outbound=T_outbound,
                                            T_inbound=T_inbound,
                                            noise=noise,
                                            cx=cx,
                                            route=(h[:T_outbound], v[:T_outbound]))

