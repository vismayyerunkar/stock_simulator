import { Injectable } from '@angular/core';
import { Socket } from 'ngx-socket-io';  

@Injectable({
	providedIn: 'root'
})

export class SocketService {
	constructor(private socket: Socket) { }

	// emit event
	getStockData(data:any) {
		this.socket.emit('GET_STOCK_DATA',{symbols:data});
	} 

	subscribeToContinousData(){
		return this.socket.fromEvent("PRICE_CHANGED")
	}

	getStaticStockData(){
		return this.socket.fromEvent("STATIC_STOCK_DATA");
	}

	fetchTopStocks() {
		// listen event
		return this.socket.fromEvent('TRENDING_STOCKS')
	}

	fetchTopCryptos(){
		return this.socket.fromEvent("TRENDING_CRYPTOS");
	}
}