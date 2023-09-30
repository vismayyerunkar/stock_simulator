import StockSocket from "stocksocket";

var stocktickers = ["TCS.NS", "IRCTC.NS"];

StockSocket.addTickers(stocktickers, stockPriceChanged);

function stockPriceChanged(data) {
  console.log(data);
}

//For single Stock
// StockSocket.addTicker("TCS.NS", stockPriceChanged);

// function stockPriceChanged(data) {
//   // Choose what to do with your data as it comes in.
//   console.log(data);
// }
