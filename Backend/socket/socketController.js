import stocksocket from "stocksocket";
import { isMarketOpen } from "../controllers/stockController.js";
import { NseIndia } from "stock-nse-india";
import { getTopCryptos } from "./crypto.js";
const nseIndia = new NseIndia()




export class StockDataHandler {
    // Maintaining the client to emit the socket events
    socket = null;
    constructor(socketClient) {
        console.log("Client connected");
        this.socket = socketClient;
    }

    //Getting live data for list of specified stocks
    GetStockDataStream = async(payload, cb) => {
        console.log("getting socket data stream")
        try {
            var stocktickers = payload.symbols;
            console.log("stock tickers",stocktickers)
            const is_market_open = await isMarketOpen();
            console.log(is_market_open)
            if (is_market_open) {
                stocktickers.forEach((symbolData) => {
                    this.socket.join(symbolData?.toUpperCase());
                });
                console.log("market is open");
                // cb({
                //     status: 200,
                //     message: "Market is open hence subscribed to the stock"
                // })
            } else {
                console.log("market is closed");
                getStockData(stocktickers).then((data) => {
                    // console.log("received data from api as market is closed : ",data)
                    const transformedData = [];
                    if (data?.length > 0) {
                        data?.forEach(stock => {
                            const rawData = stock;
                            transformedData.push({
                                id: rawData?.info?.symbol,
                                price: rawData?.priceInfo?.lastPrice,
                                dayVolume: rawData?.preOpenMarket?.totalTradedVolume
                            });
                        });
                    }
                    console.log("successfully subscribed to the stocks")
                    console.log(transformedData)
                   this.socket?.emit("STATIC_STOCK_DATA",transformedData)
                }).catch((err) => {
                    console.log("Something went wrong while fetching stock data when the market is closed")
                })
            }
        } catch (er) {
            console.log("error occured while getting stock stream")
        }

        async function getStockData(symbols) {
            if (symbols.length == 0) return [];
            const responses = [];

            for (const symbol of symbols) {
                try {
                    if (!symbol) {
                        return;
                        // throw new Error("No Symbol specified");
                    }

                    const response = await nseIndia.getEquityDetails(symbol);
                    responses.push(response);
                } catch (error) {
                    console.error(`Error fetching data for ${symbol}`);
                }
            }

            return responses;
        }
    }
}


