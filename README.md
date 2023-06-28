1- Remove faulty frames with `fill_in_faulty_frames.py`
2- Convert to polar and reduce dimension with `transform_and_reduce.py`
3- Find separate components with `identify_indpendent_regions_of_interest.py`
4- Reduce each region of interest to single signal with `reduce_component_to_signal.py`
5- Run the beamforming based speed estimation `speed_estimation_on_reduced_signals.py`
6- Run `fault_estimation_from_speed.py` to diagnose the fault