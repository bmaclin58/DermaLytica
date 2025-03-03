from DermaLytica.Prediction_Model.Marie import AGE_MEAN, AGE_STD
import numpy as np

def prepare_metadata(age, gender, location):
	"""
	Prepare the metadata array for model input

	Format: [Age, Gender_female, Gender_male, Location_Back, Location_Front Torso, Location_Head & Neck,
				 Location_Legs, Location_Mouth & Groin, Location_Palms & Soles, Location_Shoulders & Arms,
				 Location_Side Torso (Ribs)]
	"""
	# Standardize age
	age_standardized = (float(age) - AGE_MEAN) / AGE_STD

	# Initialize metadata array
	metadata = np.zeros(11, dtype = np.float32)
	metadata[0] = age_standardized

	# Set gender (one-hot encoding)
	if gender.lower() == 'female':
		metadata[1] = 1.0
	else:
		metadata[2] = 1.0

	# Set location (one-hot encoding)
	location_mapping = {
			'Back':              3,
			'Front Torso':       4,
			'Head & Neck':       5,
			'Legs':              6,
			'Mouth & Groin':     7,
			'Palms & Soles':     8,
			'Shoulders & Arms':  9,
			'Side Torso (Ribs)': 10
			}

	if location in location_mapping:
		metadata[location_mapping[location]] = 1.0

	return metadata
