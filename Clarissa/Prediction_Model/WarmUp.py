import io
import os

from PIL import Image

from Clarissa.Prediction_Model.Clarissa import clarissa_Prediction

WARMED_FLAG = "/tmp/Clarissa_hf_warmed.txt"

def warm_hf_endpoint_once():
	if os.path.exists(WARMED_FLAG):
		return

	resCode = warm_hf_endpoint()
	resCode = str(resCode)

	with open(WARMED_FLAG, "w") as f:
		f.write(resCode)


def warm_hf_endpoint():
	try:
		# Create a tiny 1x1 dummy image
		img = Image.new("RGB", (1, 1), color = "black")
		buf = io.BytesIO()
		img.save(buf, format = "PNG")

		response = clarissa_Prediction(buf.getvalue())

		print(f"HF endpoint warm-up status: {response.status_code}")
		return response.status_code

	except Exception as e:
		print(f"HF endpoint warm-up failed: {e}")
