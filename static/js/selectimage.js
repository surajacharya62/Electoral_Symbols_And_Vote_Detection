function displaySelectedImage(event, elementId) {
    const selectedImage = document.getElementById(elementId);
    const fileInput = event.target;

    if (fileInput.files && fileInput.files[0]) {
        const reader = new FileReader();

        reader.onload = function(e) {
            selectedImage.src = e.target.result;
        };

        reader.readAsDataURL(fileInput.files[0]);
    }
}


document.getElementById('uploadForm').addEventListener('submit', function(e) {
    const fileInput = document.getElementById('customFile');
    const errorAlert = document.getElementById('errorAlert');
    
    if (!fileInput.files.length) {
        e.preventDefault();
        errorAlert.textContent = 'Please select an image before predicting.';
        errorAlert.classList.remove('d-none');
        
        // Add visual feedback to the upload button
        document.getElementById('fileLabel').classList.add('btn-error');
        
        // Set focus for accessibility
        fileInput.focus();
    }
});