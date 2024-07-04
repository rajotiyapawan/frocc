import openml
import pandas as pd
import numpy as np

dataset_ids = [15,29,31,37,50,1063,1464,1480,1510,6332, 23381, 40994]
results = []

for ids in dataset_ids : 
	dataset = openml.datasets.get_dataset(ids)
	X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)

	# Separate numerical and categorical features
	N = X.select_dtypes(include=[np.number]).values
	C = X.select_dtypes(exclude=[np.number]).values
	N = N if N.shape[1] > 0 else None
	C = C if C.shape[1] > 0 else None

	# Create the info dictionary
	results.append({
		'dataset id': ids,
	    'n_classes': len(y.unique()),
	   	'n_num_features': N.shape[1] if N is not None else 0,
	    'n_cat_features': C.shape[1] if C is not None else 0,
	    'dataset_name': dataset.name,
	    'training samples': len(X),
	    'label samples': len(y)
	})

# Save results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv('results/dataset_info.csv', index=False)
