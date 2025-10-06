// In static/js/app.js (Fully Refactored)

document.addEventListener('DOMContentLoaded', () => {
    // --- WebSocket Connection ---
    const socket = io();
    socket.on('connect', () => console.log('Connected to server!'));
    socket.on('disconnect', () => console.log('Disconnected from server.'));

    // --- DOM Element References ---
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const context = canvas.getContext('2d');
    const statusMessage = document.getElementById('status-message');
    
    // Buttons and Status
    const startSessionBtn = document.getElementById('start-session-btn');
    const endSessionBtn = document.getElementById('end-session-btn');
    const saveStatus = document.getElementById('save-status');

    // UI elements for analysis data
    const emotionText = document.getElementById('emotion-text');
    const healthScore = document.getElementById('health-score');
    const fatigueCard = document.getElementById('fatigue-card');
    const fatigueAlert = document.getElementById('fatigue-alert');
    const recommendationsList = document.getElementById('recommendations-list');

    // --- State Management ---
    let sessionActive = false;
    let frameSenderInterval;
    let sessionData = {}; // Will be reset for each session
    let localStream = null;

    // --- Core Functions ---

    function startSession() {
        console.log("Starting a new session...");
        
        // Reset session data
        sessionData = { scores: [], emotions: {}, fatigueEvents: 0 };
        saveStatus.innerText = '';

        // Request webcam access
        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
            navigator.mediaDevices.getUserMedia({ video: true })
                .then(stream => {
                    localStream = stream; // Store the stream
                    video.srcObject = stream;
                    video.play();
                    statusMessage.style.display = 'none';
                    sessionActive = true;
                    
                    // Start sending frames to the server
                    frameSenderInterval = setInterval(sendFrame, 100);

                    // Update button states
                    startSessionBtn.style.display = 'none';
                    endSessionBtn.style.display = 'inline-block';
                    endSessionBtn.disabled = false;
                })
                .catch(err => {
                    console.error("Error accessing webcam:", err);
                    statusMessage.innerText = "Could not access webcam. Please grant permission.";
                });
        }
    }

    function endSession() {
        if (!sessionActive) return;
        console.log("Ending session and saving data...");

        // Stop sending frames and disable the button
        sessionActive = false;
        clearInterval(frameSenderInterval);
        endSessionBtn.disabled = true;
        saveStatus.innerText = 'Analyzing and saving summary...';

        // Stop the webcam tracks
        if (localStream) {
            localStream.getTracks().forEach(track => track.stop());
            video.srcObject = null;
        }

        // Calculate final summary
        const avgScore = sessionData.scores.length > 0
            ? Math.round(sessionData.scores.reduce((a, b) => a + b, 0) / sessionData.scores.length)
            : 'N/A';
        const dominantEmotion = Object.keys(sessionData.emotions).length > 0
            ? Object.keys(sessionData.emotions).reduce((a, b) => sessionData.emotions[a] > sessionData.emotions[b] ? a : b)
            : 'N/A';

        const summary = {
            avgScore: avgScore,
            dominantEmotion: dominantEmotion,
            fatigueEvents: sessionData.fatigueEvents
        };

        // Send summary to the server
        console.log("Sending summary to server:", summary);
        socket.emit('save_session', summary);
        
        // Update button states
        endSessionBtn.style.display = 'none';
        startSessionBtn.style.display = 'inline-block';
    }

    function sendFrame() {
        if (!sessionActive) return;
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        const dataURL = canvas.toDataURL('image/jpeg', 0.7);
        socket.emit('frame', dataURL);
    }

    function aggregateData(data) {
        if (typeof data.healthScore === 'number') sessionData.scores.push(data.healthScore);
        const emotion = data.emotion;
        sessionData.emotions[emotion] = (sessionData.emotions[emotion] || 0) + 1;
        if (data.fatigueAlert) sessionData.fatigueEvents++;
    }

    function updateUI(data) {
        // ... (The updateUI function from the previous step remains the same) ...
        if (emotionText.textContent !== data.emotion) {
            emotionText.textContent = data.emotion;
            emotionText.classList.add('pulse');
            setTimeout(() => emotionText.classList.remove('pulse'), 500);
        }
        if (healthScore.textContent !== data.healthScore.toString()) {
            healthScore.textContent = data.healthScore;
            healthScore.classList.add('pulse');
            setTimeout(() => healthScore.classList.remove('pulse'), 500);
        }
        fatigueAlert.textContent = data.fatigueAlert ? 'YES' : 'NO';
        fatigueCard.classList.toggle('alert', data.fatigueAlert);
        recommendationsList.innerHTML = '';
        data.recommendations.forEach(rec => {
            const li = document.createElement('li');
            li.textContent = rec;
            recommendationsList.appendChild(li);
        });
    }

    // --- Event Listeners ---
    
    // Listen for server confirmation of saved session
    socket.on('session_saved', (response) => {
        if (response.status === 'success') {
            saveStatus.innerText = 'Session saved successfully!';
        } else {
            saveStatus.innerText = `Error saving session: ${response.message}`;
            console.error("Server failed to save session:", response.message);
        }
    });

    // Listen for data from the server and update UI
    socket.on('analysis_results', (data) => {
        if (!sessionActive) return;
        updateUI(data);
        aggregateData(data);
    });

    // Wire up the buttons
    startSessionBtn.addEventListener('click', startSession);
    endSessionBtn.addEventListener('click', endSession);

    // --- Initial Call ---
    startSession(); // Automatically start the first session on page load
});