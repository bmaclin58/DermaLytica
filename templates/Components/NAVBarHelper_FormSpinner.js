document.addEventListener('DOMContentLoaded', function () {
	// Get references to form and loading elements
	const analysisForm = document.getElementById('analysisForm');
	const loadingIndicator = document.getElementById('loadingIndicator');

	// Ensure the form exists in the DOM
	if (analysisForm) {
		analysisForm.addEventListener('submit', function (event) {
			console.log('Form is being submitted...');

			// Show the loading indicator
			loadingIndicator.classList.remove('d-none');

			// Disable the submit button and update its text
			const submitButton = analysisForm.querySelector('button[type="submit"]');
			if (submitButton) {
				submitButton.disabled = true;
				submitButton.textContent = 'Processing...';
			}
		});
	}

	// Hover interaction for help bubbles
	const cardBody = document.querySelector('.card-body');
	const formHelp = document.querySelector('.form-help');
	if (cardBody && formHelp) {
		cardBody.addEventListener('mouseenter', function () {
			formHelp.classList.remove('d-none');
		});
		cardBody.addEventListener('mouseleave', function () {
			formHelp.classList.add('d-none');
		});
	}

	// Help bubbles for navigation items
	const navItems = document.querySelectorAll('.nav-item');
	navItems.forEach(item => {
		const helpBubble = item.querySelector('.help-bubble');
		if (helpBubble) {
			item.addEventListener('mouseenter', function () {
				helpBubble.classList.remove('d-none');
			});
			item.addEventListener('mouseleave', function () {
				helpBubble.classList.add('d-none');
			});
		}
	});
});
