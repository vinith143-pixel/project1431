const url = 'https://trading-view.p.rapidapi.com/market/get-movers?exchange=US&name=volume_gainers&locale=en';
const options = {
    method: 'GET',
    headers: {
        'x-rapidapi-key': 'a9a2f6b810mshb4d15524c9b67ffp10f75fjsnb8286e6cf2a0',
        'x-rapidapi-host': 'trading-view.p.rapidapi.com'
    }
};

async function fetchData() {
    const resultContainer = document.getElementById('result');
    const errorContainer = document.getElementById('error');

    try {
        const response = await fetch(url, options);

        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        const data = await response.json();

        // Build the HTML table
        let tableHTML = `
            <table>
                <thead>
                    <tr>
                        <th>Symbol</th>
                        <th>Volume</th>
                    </tr>
                </thead>
                <tbody>
        `;

        data.symbols.forEach(symbol => {
            tableHTML += `
                <tr>
                    <td>${symbol.s}</td>
                    <td>${symbol.f[0].toLocaleString()}</td>
                </tr>
            `;
        });

        tableHTML += `
                </tbody>
            </table>
        `;

        // Insert the table into the result container
        resultContainer.innerHTML = tableHTML;
    } catch (error) {
        console.error('Error:', error);
        errorContainer.textContent = `Error fetching data: ${error.message}`;
    }
}

// Call the fetchData function when the script loads
fetchData();
