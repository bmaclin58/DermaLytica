import os

from BullyFilter.Prediction_Model.BullyFilterModel import bullyfilter_request

WARMED_FLAG = "/tmp/BullyFilter_hf_warmed.txt"


def warm_hf_endpoint_once() -> None:
	if os.path.exists(WARMED_FLAG):
		return

	status = warm_hf_endpoint()
	with open(WARMED_FLAG, "w") as f:
		f.write(str(status))


def warm_hf_endpoint() -> int | None:
	try:
		resp = bullyfilter_request("Wake", max_new_tokens = 5, timeout = 20)
		# don't raise; warmup should be best-effort
		return resp.status_code
	except Exception as e:
		print(f"HF endpoint warm-up failed: {e}")
		return None
