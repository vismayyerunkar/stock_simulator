import { SocketService } from 'src/services/socketService';
import { Component, AfterViewInit, OnInit } from '@angular/core';
import { ActivatedRoute } from '@angular/router';
import { AssetService } from 'src/services/asset.service';
import axios from 'axios';
import { environment } from 'src/environments/environment';
import { webSocket } from 'rxjs/webSocket';
import { MessageService, SelectItem } from 'primeng/api';
declare const TradingView: any;

@Component({
  selector: 'app-stock-details',
  templateUrl: './stock-details.component.html',
  styleUrls: ['./stock-details.component.scss'],
})
export class StockDetailsComponent implements AfterViewInit {
  quantity: number = 1;
  subtotal: number;
  title: string; // Add this property to store the title
  ASSET_TYPE: string;
  cryptoSocket: any = undefined;

  //For Buy
  assetSymbol: string;
  assetName: string;
  assetPrice: number;
  assetType: string = 'STOCK';
  assetQuantity: number;
  cryptoMap = new Map();

  currentStock: any;

  //WatchList
  stockSymbol: string;
  stockName: string;
  type: string = 'STOCK';
  //Added explisitly to check

  // subscribeToWebsocket() {
  //   const subject: any = webSocket('wss://streamer.finance.yahoo.com');

  //   subject.subscribe((res: any) => {
  //     console.log('Response from websocket: ' + res);
  //   });
  //   console.log('websocket subscribed');
  //   subject.next({ subscribe: ['IRCTC'] });
  // }

  async createCryptoMap() {
    const response = await axios.get('https://api.coincap.io/v2/assets');

    for (let i = 0; i < response.data.data.length; i++) {
      this.cryptoMap.set(
        response.data?.data[i]?.id,
        response.data?.data[i]?.symbol
      );
    }
  }

  constructor(
    private route: ActivatedRoute,
    private buyreq: AssetService,
    private socketService: SocketService,
    private messageService: MessageService
  ) {
    // Retrieve the title parameter from the route

    this.route.params.subscribe((params: any) => {
      this.title = params.title;
      console.log(params);
      this.ASSET_TYPE = params?.type;
    });

    // this.subscribeToWebsocket();

    socketService.subscribeToContinousData().subscribe((data: any) => {
      console.log('socket live data : ', data);

      if (data?.id?.split('.')[0]?.toLowerCase() == this.title.toLowerCase())
        this.currentStock = data;
      this.subtotal = this.currentStock?.price * this.quantity;
    });

    console.log(this.title);

    console.log(this.ASSET_TYPE);
    if (this.ASSET_TYPE == 'STOCK') {
      socketService.getStockData([this.title]);
      socketService.getStaticStockData()?.subscribe((data: any) => {
        console.log('static stock data : ', data);
        this.currentStock = data[0];
        this.subtotal = data[0].price;
        this.assetPrice = data[0]?.price;
      });
    } else if (this.ASSET_TYPE == 'CRYPTO') {
      if (this.ASSET_TYPE === 'CRYPTO') {
        console.log('established socketiopd');
        this.cryptoSocket = new WebSocket(
          `wss://ws.coincap.io/prices?assets=${this.title.toLowerCase()}`
        );
      }

      if (this.cryptoSocket) {
        const temptite = this.title;

        console.log(temptite);
        this.cryptoSocket.onmessage = (msg: any) => {
          console.log(JSON.parse(msg.data));
          const objCopy = {
            price: Object.entries(JSON.parse(msg.data))[0][1],
            id: temptite,
          };
          this.currentStock = objCopy; // Update the class variable
          this.subtotal = this.currentStock?.price * this.quantity;
        };
      }
    } else {
      console.log('invalid asset name');
    }
  }

  ngAfterViewInit(): void {
    this.createCryptoMap().then(() => {
      this.loadTradingViewLibrary();
    });
  }

  loadTradingViewLibrary() {
    const script = document.createElement('script');
    script.src = 'https://s3.tradingview.com/tv.js';
    script.async = true;
    script.onload = () => {
      this.initializeTradingViewWidget();
    };
    document.head.appendChild(script);
  }

  initializeTradingViewWidget() {
    if (typeof TradingView !== 'undefined') {
      new TradingView.widget({
        with: '100%',
        symbol:
          this.cryptoMap.get(this.title.toLocaleLowerCase()) ?? this.title,
        interval: 'D',
        timezone: 'Etc/UTC',
        theme: 'light',
        style: '1',
        locale: 'en',
        toolbar_bg: '#f1f3f6',
        enable_publishing: false,
        allow_symbol_change: true,
        container_id: 'trading-chart',
      });
    } else {
      console.error('TradingView library is not loaded.');
    }
  }
  calculateSubtotal() {
    // Calculate the subtotal based on the quantity
    this.subtotal = this.quantity * this.currentStock?.price; // Replace with the actual stock price
  }

  //this function will accept a name of stock
  buyAssest() {
    console.log(this.currentStock);
    axios.interceptors.request.use(function (config) {
      config.headers.Authorization = `Bearer ${localStorage.getItem(
        'authToken'
      )}`;
      return config;
    });
    axios
      .post(`${environment.baseUrl}/api/assets/purchaseAsset`, {
        assetSymbol: this.title.toUpperCase(),
        assetName: this.title.toUpperCase(),
        assetPrice: this.currentStock.price,
        assetType: this.ASSET_TYPE,
        assetQuantity: this.quantity,
      })
      .then(function (response) {
        console.log(response);
        alert('Buy successfull');
      })
      .catch(function (error) {
        console.log(error);
        alert('Something went wrong , please try again');
      });

    return;
    // Create a FormData object with your data
  }

  addToWatchlist() {
    axios.interceptors.request.use(function (config) {
      config.headers.Authorization = `Bearer ${localStorage.getItem(
        'authToken'
      )}`;
      return config;
    });
    axios
      .post(`${environment.baseUrl}/api/watchlist/createWatchList`, {
        stockSymbol: this.title.toUpperCase(),
        stockName: this.title.toUpperCase(),
        type: this.assetType,
      })
      .then(function (response) {
        console.log(response);
        // this.messageService.add({
        //   severity: 'success',
        //   summary: 'Added To the Watchlist',
        //   detail: 'Richin',
        // });
        alert('Added To the Watchlist');
      })
      .catch(function (error) {
        console.log(error);
        alert('Something went wrong , please try again');
      });

    return;
  }
}
