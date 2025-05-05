document.getElementById('analyzeButton').addEventListener('click', async () => {
    const jobDescription = document.getElementById('jobDescription').value.trim();
    const resumeFile = document.getElementById('resumeUpload').files[0];
    const analysisType = document.querySelector('input[name="analysisType"]:checked').value;
    const analyzeButton = document.getElementById('analyzeButton');
    const resultsContainer = document.getElementById('analysisResults');
    const resultsText = document.getElementById('results');

    // Validation
    if (!resumeFile) {
        alert('Please upload a resume file.');
        return;
    }
    if (!jobDescription) {
        alert('Please provide the job description.');
        return;
    }
    if (resumeFile.type !== 'application/pdf') {
        alert('Only PDF files are supported.');
        return;
    }

    // Show loading state
    analyzeButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
    analyzeButton.disabled = true;
    resultsText.textContent = '';

    const formData = new FormData();
    formData.append('resume', resumeFile);
    formData.append('job_description', jobDescription);
    formData.append('analysis_type', analysisType);

    try {
        const response = await fetch('/analyze', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        if (data.error) {
            alert(data.error);
        } else {
            resultsText.textContent = data.analysis;
            resultsContainer.classList.remove('hidden');
        }
    } catch (error) {
        alert('An error occurred while processing the analysis. Please try again.');
    } finally {
        // Reset button state
        analyzeButton.innerHTML = '<i class="fas fa-search"></i> Analyze Resume';
        analyzeButton.disabled = false;
    }
});

document.getElementById('exportButton').addEventListener('click', async () => {
    const analysis = document.getElementById('results').textContent;
    
    if (!analysis) {
        alert('No analysis data to export.');
        return;
    }

    try {
        const response = await fetch('/export', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ analysis })
        });

        if (response.ok) {
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'resume_analysis.txt';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
        } else {
            alert('Error exporting analysis.');
        }
    } catch (error) {
        alert('An error occurred while exporting the analysis.');
    }
});
