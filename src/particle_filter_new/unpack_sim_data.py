import numpy as np

d = np.load('rod_results.npz')['human_handoff_ik_solutions'].item()
full_dataset = []
full_dataset.extend(list(d['Rod right']))
full_dataset.extend(list(d['Rod left']))
# full_dataset.extend(list(d['Mug side']))
# full_dataset.extend(list(d['Mug top']))
# full_dataset.extend(list(d['Mug side upside down']))
# full_dataset.extend(list(d['Mug handle']))
# full_dataset.extend(list(d['Mug bottom']))
# full_dataset.extend(list(d['Mug handle upside down']))
dataset = np.array(full_dataset)
np.save("sim_data_rod", dataset)
