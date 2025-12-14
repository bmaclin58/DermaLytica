import io
import os

from PIL import Image

from BullyFilter.Prediction_Model.BullyFilterModel import BullyFilter_Prediction

WARMED_FLAG = "/tmp/BullyFilter_hf_warmed.txt"

def warm_hf_endpoint_once():
	if os.path.exists(WARMED_FLAG):
		return

	resCode = warm_hf_endpoint()
	resCode = str(resCode)

	with open(WARMED_FLAG, "w") as f:
		f.write(resCode)


def warm_hf_endpoint():
	try:
		response = BullyFilter_Prediction(
				{
						"inputs"    : "Wake",
						"parameters": {
								"max_new_tokens": 5
								}
						}
				)

		print(f"HF endpoint warm-up status: {response.status_code}")
		return response.status_code

	except Exception as e:
		print(f"HF endpoint warm-up failed: {e}")
