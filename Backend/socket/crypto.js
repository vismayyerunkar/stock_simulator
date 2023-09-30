import axios from "axios";

export async function getTopCryptos(n) {
    try {
      // Make a GET request to the CoinGecko API to fetch the top 5 cryptocurrencies
      const response = await axios.get(
        'https://api.coincap.io/v2/assets'
      );
      const gainers = [];
      const loosers = [];

      const top5Cryptos = response?.data.data;
      top5Cryptos?.sort((a,b)=>b?.priceUsd - a?.priceUsd);

      for(let i = 0;i<n;i++){
        gainers.push(top5Cryptos[i]);
      }

      for(let j = top5Cryptos?.length -1 ;j>=top5Cryptos?.length - n;j--){
        loosers.push(top5Cryptos[j]);
      }

        // Print the top 5 cryptocurrencies
        return {
            gainers:gainers,
            loosers:loosers
        };

    
    } catch (error) {
      console.error('Error fetching top crypto data');
    }
  }
  
  // Call the function to get the top 5 cryptocurrencies
 
  
  
  
  