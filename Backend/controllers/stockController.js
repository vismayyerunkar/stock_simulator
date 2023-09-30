import axios from "axios";
import { NseIndia } from "stock-nse-india";
const nseIndia = new NseIndia()

export const getCurrentStockPrice = async(symbol) => {
    console.log("getting price")
    try {
        const resp = await nseIndia.getEquityDetails(symbol);
        return resp.priceInfo?.lastPrice;
    } catch (err) {
        throw Error(err);
    }
}


export const getStockPriceBetweenDateRange = async(req, res) => {

    const { symbol, start, end } = req.body;

    try {
        const range = {
            start: new Date(start),
            end: new Date(end)
        }

        const data = await nseIndia.getEquityHistoricalData(symbol, range);
        console.log(data[0].data);
        return res.send({ data: data });

    } catch (err) {
        throw Error(err);
    }
}




export const getGainersAndLoosers = async(limit) => {

    try{

    const indexData = await nseIndia.getEquityStockIndices("NIFTY 50");

    const gainers = [];
    const losers = [];
    indexData?.data?.forEach((equityInfo) => {
        if (gainers.length < limit && equityInfo.pChange > 0)
            gainers.push(equityInfo)
        else if (losers.length < limit && equityInfo.pChange <= 0) {
            losers.push(equityInfo)
        } else {
            return true;
        }
    });

    const data = {
        gainers: [...gainers].sort((a, b) => b.pChange - a.pChange),
        losers: [...losers].sort((a, b) => a.pChange - b.pChange)
    }
    return data;
}catch(err){
    console.log("Error occured while fetching top gainers and loosers")
}

}

export const isMarketOpen = async() => {
    try {

        const data = await axios.get("https://www.nseindia.com/api/marketStatus");
        const marketStatus = data.data.marketState[0].marketStatus;
        if (marketStatus.toLowerCase() == 'closed' || marketStatus.toLowerCase() == 'close') {
            return false;
        }
        return true;
    } catch (err) {
        console.log("Failed to fetch market status : ", err)
    }
}