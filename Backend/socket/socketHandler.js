import { Server } from "socket.io";
import { StockDataHandler } from "./socketController.js";
import { getGainersAndLoosers, isMarketOpen } from "../controllers/stockController.js";
import stocksocket from "stocksocket"
import CoinGecko from 'coingecko-api';
import axios from "axios";
import { getTopCryptos } from "./crypto.js";

async function getCryptoSymbolsAndNames() {
    try {
        const response = await axios.get('https://api.coingecko.com/api/v3/coins/list');
        const cryptoData = response.data;

        const cryptoSymbolsToNames = {};

        cryptoData.forEach((crypto) => {
            cryptoSymbolsToNames[crypto.symbol.toUpperCase()] = crypto.name;
        });

        return cryptoSymbolsToNames;
    } catch (error) {
        console.error('Error fetching cryptocurrency data:', error);
        return {};
    }
}

const createSocketServer = (server) => {
    const io = new Server(server, {
        cors: {
            origin: '*',
            methods: ['GET', 'POST'],
        },
    });
    return io;
}
let io;
export const handleSocket = (server) => {
    io = createSocketServer(server);
    console.log("Socket connection established")
    listenSocketEvents(io);
}

var userToCroptosMap = {};

function addValueToKey(key, value) {
    // Shorcut || returns left side if it is "truthy," or the right otherwise.
    // This means that we only assign a new Array to the Object's property
    // if it has not previously been used.
    userToCroptosMap[key] = userToCroptosMap[key] || [];
    // Adds a value to the end of the Array
    userToCroptosMap[key].push(value);
}


function getRequestedCryptos(key) {
    return userToCroptosMap[key];
}


const getCryptoData = async(names, symbols) => {

    const CoinGeckoClient = new CoinGecko();
    let data = await CoinGeckoClient.exchanges.fetchTickers('bitfinex', {
        coin_ids: ['bitcoin', 'ethereum', 'ripple', 'litecoin', 'stellar', ...names]
    });

    console.log(data.data.tickers);
    var _coinList = {};
    var _datacc = data.data.tickers.filter(t => t.target == 'USD');
    [
        'BTC',
        'ETH',
        'XRP',
        'LTC',
        'XLM',
        ...symbols
    ].forEach((i) => {
        var _temp = _datacc.filter(t => t.base == i);
        var _res = _temp.length == 0 ? [] : _temp[0];
        _coinList[i] = _res.last;
    })
    console.log(_coinList);
    return _coinList;
}




// 1) when the client needs a particular stock price He well join the particular socket room
// 2) if market is open then for all the room names (stock symbols) we will listen to the price change else we will return the last price of the stock
// 3) when the price changes we will emit the new stock data to that room
// 4) when the client does not want the stock data just leave the room


function sleep(millis){
 
    var date = new Date();
    var curDate = null;
    do { curDate = new Date(); }
    while(curDate-date < millis);
}

const temptickers = []

const filter= (string)=>{
    for(let i = 0;i<string?.length;i++){
        if(/\d/.test(string[i])){
            return false
        }
    }

    return string;
}

setInterval(async () => {
    const tickers = io.of("/").adapter.rooms;
    const marketStatus = await isMarketOpen();
    if(!marketStatus){
        return;
    }

    for (let [key, value] of tickers) {
        const filteredData = filter(key)
        if(filteredData && !temptickers.includes(filteredData+".NS")){
            temptickers.push(filteredData?.toUpperCase() + ".NS");
        }
    }

    console.log("temp tickers :",temptickers)
    if (temptickers.length > 0 && marketStatus) {
        stocksocket.removeAllTickers();
        sleep(2000);
        stocksocket.addTickers(temptickers, (newPrice) => {
            console.log("Price changed for : ", newPrice);
            io.to(newPrice.id?.split(".")[0]).emit("PRICE_CHANGED", newPrice);
        });
    }else{
        // console.log("market is close")
    }
}, 10000)



const listenSocketEvents = (io) => {

    try {
        console.log(io.of("/").adapter.rooms)


        io.on("connection", (socket) => {

            const handler = new StockDataHandler(socket);
            // listening to events 
            socket.on("GET_STOCK_DATA", (payload, cb) => {
                handler.GetStockDataStream(payload, cb)
            });

            // socket.on("SUBSCRIBE_CRYPTOS", (payload, cb) => addValueToKey(payload.UID, payload.cryptos))
            // socket.on("UNSUBSCRIBE_CRYPTOS", (payload, cb) => delete userToCroptosMap[payload.UID]);
            socket.on('disconnect', () => {
                console.log('client disconnected');
            });
        });

        // Getting the list of trending stocks
        let timeout = setInterval(async() => {
            io.sockets.emit("TRENDING_STOCKS", await getGainersAndLoosers(7));
            io.sockets.emit("TRENDING_CRYPTOS", await getTopCryptos(6));
        }, 5000);

        io.on("disconnect", () => {
            clearInterval(timeout);
            console.log('client disconnected');
        });
    } catch (err) {
        console.log("Error while socket client connection")
    }
}

