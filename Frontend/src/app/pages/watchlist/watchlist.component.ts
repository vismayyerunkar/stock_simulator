import { webSocket } from 'rxjs/webSocket';
import { Component, OnInit } from '@angular/core';
import axios from 'axios';

import { SelectItem } from 'primeng/api';
import { environment } from 'src/environments/environment';
import { Stock } from 'src/models/stock';
import { IWatchListResponse, Watchlist } from 'src/models/watchlist';
import { SocketService } from 'src/services/socketService';
import { StockService } from 'src/services/stock.service';

@Component({
  selector: 'app-watchlist',
  templateUrl: './watchlist.component.html',
  styleUrls: ['./watchlist.component.scss'],
})
export class WatchlistComponent implements OnInit {
  stocks: Stock[] = [];
  filteredStocks: Stock[] = [];
  sortOptions: SelectItem[];
  watchlist: any[];
  assests: any[];
  updatedStocks:any;
  symbols:string[] = [];

  removeWatchlist:any;
  cryptoMap:Map<string,string> = new Map();
  cryptoSocket:any


  subscribeToWebsocket(cryptoNames:string) {
    this.cryptoSocket = new WebSocket(
      `wss://ws.coincap.io/prices?assets=${cryptoNames}`
    );   
  }

  async createCryptoMap() {
    const response = await axios.get('https://api.coincap.io/v2/assets');
    for (let i = 0; i < response.data.data.length; i++) {
      this.cryptoMap.set(
        response.data?.data[i]?.id,
        response.data?.data[i]?.symbol
      );
    }
  }


  constructor(private stockService: StockService,private socketService: SocketService) {
    this.createCryptoMap().then(()=>{
      this.getWatchlistList();
    })
    
    console.log(this.cryptoMap)

    
    this.sortOptions = [
      { label: 'Symbol', value: 'symbol' },
      { label: 'Company Name', value: 'name' },
      { label: 'Price', value: 'price' },
      { label: 'Change (%)', value: 'changePercentage' },
    ];

     

    setTimeout(()=>{
      const set = new Set(this.symbols);
      socketService.getStockData([...set]);
      console.log("symbol",this.symbols);

      socketService.subscribeToContinousData().subscribe((data:any)=>{
        //setting the live price
        console.log("socket live data:",data);
       
        for(let i= 0;i<this.watchlist.length;i++){
          if(this.watchlist[i]?.stockSymbol?.toLowerCase() == data.id?.split(".")[0]?.toLowerCase()){
            this.watchlist[i] = {
              ...this.watchlist[i],
              currentPrice:data?.price,
              changePercent:data?.changePercent
            }
          }
        }
  
  
      })

      
    },2000);

   
    setTimeout(()=>{
      if(this.cryptoSocket){
        this.cryptoSocket.onmessage = (msg: any) => {
          console.log(JSON.parse(msg.data));
    
          for(let i= 0;i<this.watchlist.length;i++){
            if(this.watchlist[i]?.stockSymbol?.toLowerCase() == Object.entries(JSON.parse(msg.data))[0][0]?.toLowerCase()){
              this.watchlist[i] = {
                ...this.watchlist[i],
                currentPrice:Object.entries(JSON.parse(msg.data))[0][1], // current dollar price
                changePercent: this.calculatePercentageChange(this.watchlist[i].currentPrice ,(Object.entries(JSON.parse(msg.data))[0][1] as number))
              }
            }
          }
          console.log("crypto live data : ")
        };
      }
      
    },3000)
    
    
    socketService.getStaticStockData()?.subscribe((data:any)=>{
      console.log("static stock data : ",data);

      const map = new Map();

      data?.forEach((ele:any) => {
          map.set(ele?.id,ele?.price);
      });

      const copy:any[] = [];

      this.watchlist.map((Wassest:any)=>{
        const entry = {
          ...Wassest,
          currentPrice:map.get(Wassest?.stockSymbol)
        }
        copy.push(entry);
      });
  
      this.watchlist = copy;
      console.log(this.watchlist);
    })

  }


  calculatePercentageChange(oldValue:number, newValue:number) {
    var change = newValue - oldValue;
  
    var absoluteOldValue = Math.abs(oldValue);
  
    var percentageChange = (change / absoluteOldValue) * 100;
  
    return percentageChange;
  }


  

  ngOnInit(): void {
    
  }
  createNewWatchlist(): void {}

  getWatchlistList() {
    this.stockService.fetchWatchList().subscribe({
      next: (res: any) => {
        this.watchlist = res;
        console.log(res);

        if(typeof this.watchlist == typeof []){
          this.watchlist.reverse();
        }


        let cryptos = "";
        console.log(this.cryptoMap)
        const temp:any[] = []
          res?.forEach((d:any)=>{
            temp?.push(d?.stockSymbol);
            if(this.cryptoMap.get(d?.stockSymbol?.toLowerCase())){
                cryptos += d?.stockSymbol.toLowerCase() + ","
            }
          })

          if(cryptos[cryptos.length-1] == ","){
            cryptos = cryptos.slice(0,cryptos.length - 1);
          }
          this.symbols = temp;
          console.log(cryptos);
          this.subscribeToWebsocket(cryptos);
      },
    });
  }

  // deleteStock(watchlistId: string): void {
  //   // Send a request to your backend API to remove the watchlist by its ID
  //   this.stockService.removeWatchList(watchlistId).subscribe({
  //     next: (response: any) => {
  //       console.log('Watchlist removed successfully:', response.message);
  
  //       // Remove the deleted watchlist from the frontend array
  //       this.watchlist = this.watchlist.filter((watchlist) => watchlist.id !== watchlistId);
  //     },
  //     error: (error: any) => {
  //       console.error('Error removing watchlist:', error);
  //     },
  //   });
  // }

  deleteWatchlist(watchlistId : string) {
    console.log(watchlistId);
    axios.interceptors.request.use(function (config) {
      config.headers.Authorization = `Bearer ${localStorage.getItem(
        'authToken'
      )}`;
      return config;
    });
    axios
      .post(`${environment.baseUrl}/api/watchlist/removeWatchList`, {
        removeWatchlist : watchlistId
      })
      .then(function (response) {
        console.log(response);
        alert('Watchlist removed successfully');
        window.location.reload();
      })
      .catch(function (error) {
        console.log(error);
        alert('Something went wrong , please try again');
      });

    return;
  }

}
// {\"Symbol\":\"ADANIENT.NS\",
// \"cagr\":0.2012161094,\
// "marketCap\":2793106046976.0,\
// "price\":2450.0500488281,\
// "purchase\":2450.0500488281,\
// "quantity\":1.0,\
// "exp_cagr\":0.9974700818,\"
// exp_returns\":492.9895387423,\
// "percentage\":4.9572078728}