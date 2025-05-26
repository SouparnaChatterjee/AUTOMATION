let isProcessing = false;

async function uploadFile() {
    if (isProcessing) return;
    
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];
    
    if (!file) {
        alert('Please select a file first!');
        return;
    }

    isProcessing = true;
    updateStatus('Processing...', 0);

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/process', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error('Processing failed');

        const result = await response.json();
        
        if (file.type.startsWith('video/')) {
            displayVideo(result.original_url, result.processed_url);
        } else {
            displayImage(result.original_url, result.processed_url);
        }
        
        updateStatus('Processing complete!', 100);
    } catch (error) {
        console.error('Error:', error);
        updateStatus('Error during processing', 0);
    } finally {
        isProcessing = false;
    }
}

function displayVideo(originalUrl, processedUrl) {
    const originalVideo = document.getElementById('originalVideo');
    const processedVideo = document.getElementById('processedVideo');
    const originalImage = document.getElementById('originalImage');
    const processedImage = document.getElementById('processedImage');

    originalVideo.style.display = 'block';
    processedVideo.style.display = 'block';
    originalImage.style.display = 'none';
    processedImage.style.display = 'none';

    originalVideo.src = originalUrl;
    processedVideo.src = processedUrl;
}

function displayImage(originalUrl, processedUrl) {
    const originalVideo = document.getElementById('originalVideo');
    const processedVideo = document.getElementById('processedVideo');
    const originalImage = document.getElementById('originalImage');
    const processedImage = document.getElementById('processedImage');

    originalVideo.style.display = 'none';
    processedVideo.style.display = 'none';
    originalImage.style.display = 'block';
    processedImage.style.display = 'block';

    originalImage.src = originalUrl;
    processedImage.src = processedUrl;
}

function updateStatus(message, progress) {
    document.getElementById('status').textContent = message;
    document.getElementById('progress').style.width = `${progress}%`;
}
