document.addEventListener("DOMContentLoaded", function () {
	const fileInput = document.querySelector('input[type="file"]');
	const previewContainer = document.getElementById("imagePreviewContainer");
	const previewImage = document.getElementById("imagePreview");
	const form = document.getElementById("analysisForm");
	const loadingIndicator = document.getElementById("loadingIndicator");

	if (!fileInput) return;

	// Show image preview immediately after selection
	fileInput.addEventListener("change", function () {
		const file = this.files[0];

		if (!file) {
			previewContainer.classList.add("d-none");
			return;
		}

		// Only allow images
		if (!file.type.startsWith("image/")) {
			alert("Please upload a valid image file.");
			this.value = "";
			previewContainer.classList.add("d-none");
			return;
		}

		const reader = new FileReader();
		reader.onload = function (e) {
			previewImage.src = e.target.result;
			previewContainer.classList.remove("d-none");
		};
		reader.readAsDataURL(file);
	});

	// Optional: show spinner on submit
	form.addEventListener("submit", function () {
		loadingIndicator.classList.remove("d-none");
	});
});
