
const getData= async live_data =>{


    const url = 'https://real-time-finance-data.p.rapidapi.com/search?query=Apple&language=en';
const options = {
	method: 'GET',
	headers: {
		'x-rapidapi-key': 'a9a2f6b810mshb4d15524c9b67ffp10f75fjsnb8286e6cf2a0',
		'x-rapidapi-host': 'real-time-finance-data.p.rapidapi.com'
	}
};

try {
	const response = await fetch(url, options);
	const result = await response.json();
    const stocks = result.data.stock
	console.log(stocks);
    result.data.cash_flow.forEach(data => {
        
                 
                });
} catch (error) {
	console.error(error);
}
}
getData()