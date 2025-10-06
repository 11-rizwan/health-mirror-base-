// In static/js/history.js (Fully Corrected)

document.addEventListener('DOMContentLoaded', () => {
    const tableBody = document.getElementById('history-table-body');
    const chartCanvas = document.getElementById('history-chart').getContext('2d');
    let historyChart; // Variable to hold the chart instance

    // Fetch data from our API endpoint
    fetch('/api/get_history')
        .then(response => response.json())
        .then(data => {
            // This is a great spot for debugging! Let's log the data.
            console.log('Received data from server:', data);

            if (!data || data.length === 0) {
                tableBody.innerHTML = '<tr><td colspan="4">No history found. Complete a monitoring session to see your data.</td></tr>';
                return;
            }
            populateTable(data);
            createChart(data);
        })
        .catch(error => {
            console.error('Error fetching history:', error);
            tableBody.innerHTML = '<tr><td colspan="4">Could not load history data.</td></tr>';
        });

    function populateTable(data) {
        tableBody.innerHTML = ''; // Clear any existing rows
        data.forEach(log => {
            // FIX: Changed 'log.emotion' to 'log.dominantEmotion'
            const row = `
                <tr>
                    <td>${log.timestamp}</td>
                    <td>${log.dominantEmotion}</td>
                    <td>${log.avgScore}</td>
                    <td>${log.fatigueEvents}</td>
                </tr>
            `;
            tableBody.innerHTML += row;
        });
    }

    function createChart(data) {
        // Data needs to be reversed for a chronological chart (oldest to newest)
        const reversedData = data.slice().reverse();

        const labels = reversedData.map(log => new Date(log.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }));
        const scores = reversedData.map(log => log.avgScore);

        if (historyChart) {
            historyChart.destroy(); // Destroy old chart instance if it exists
        }

        historyChart = new Chart(chartCanvas, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Average Health Score',
                    data: scores,
                    borderColor: '#8a78ff', // Using your theme's primary color
                    backgroundColor: 'rgba(138, 120, 255, 0.2)', // A transparent version
                    fill: true,
                    tension: 0.4 // Makes the line curvy and smooth
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100
                    }
                }
            }
        });
    }
});