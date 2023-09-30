import { AssetService } from './../../../services/asset.service';
import { Component, OnInit } from '@angular/core';
import { map } from 'highcharts';
import { MenuItem } from 'primeng/api';
import { Product } from 'src/models/Product';
import { ProductService } from 'src/services/productservice';
import { SocketService } from 'src/services/socketService';
import { StockService } from 'src/services/stock.service';

@Component({
  selector: 'app-wallet',
  templateUrl: './wallet.component.html',
  styleUrls: ['./wallet.component.scss'],
})
export class WalletComponent implements OnInit {
  items: MenuItem[] = [];
  stocksData: any[] = [];
  products: Product[] = [];
  assets:any[]
  symbols:string[] = []
  allInvestedPrice:number = 0;
  allCurrentPrice:number = 0;
  stockInvestedPrice:number = 0;
  stockCurrentPrice:number = 0;
  cryptoInvestedPrice:number = 0;
  cryptoCurrentPrice:number = 0;
  cryptoSocket:any;
  cryptoQuantityMap:Map<string,number> = new Map();
  priceCryptoMap:Map<string,number> = new Map();
  loader:boolean = this.cryptoQuantityMap.size > this.priceCryptoMap.size;
  // get the list of stocks from the api and then get the live data for this stock if the market is open the subscribe to continous data


  calculateGainLossPercent(investedPrice:number, currentValue:number) {
    return ((currentValue - investedPrice) / investedPrice) * 100;
  }

  getContinousCryptoData = ()=>{
    this.cryptoSocket = new WebSocket(`wss://ws.coincap.io/prices?assets=${this.symbols?.join(",")}`);

    this.cryptoSocket.onmessage = (msg:any) => {
      console.log(JSON.parse(msg.data));
      const objCopy:any = {
        price: parseFloat(Object.entries(JSON.parse(msg.data))[0][1] as string),
      };

      const cryptoName = Object.entries(JSON.parse(msg.data))[0][0];
      console.log(cryptoName);

      if(!this.priceCryptoMap.get(cryptoName)){
        console.log("in iff",this.cryptoQuantityMap.get(cryptoName))
        this.priceCryptoMap.set(cryptoName,objCopy?.price);
        this.cryptoCurrentPrice += objCopy?.price * (this.cryptoQuantityMap.get(cryptoName) ?? 1)
        this.allCurrentPrice += objCopy?.price * (this.cryptoQuantityMap.get(cryptoName) ?? 1)
      }else{
        console.log("in else",this.cryptoQuantityMap.get(cryptoName))
        const previousPrice = this.priceCryptoMap.get(cryptoName);
        if(previousPrice != null || previousPrice != undefined){
          this.cryptoCurrentPrice += (objCopy?.price- previousPrice) * (this.cryptoQuantityMap.get(cryptoName) ?? 1)
          this.allCurrentPrice += (objCopy?.price- previousPrice) * (this.cryptoQuantityMap.get(cryptoName) ?? 1)
        }

        this.loader =  this.cryptoQuantityMap.size >= this.priceCryptoMap.size
      }
    };
  }

  constructor(private stockService: AssetService,private socketService: SocketService) {
    this.fetchStocks();

    setTimeout(()=>{
      this.getContinousCryptoData();
    },2000);
      
  }

  async fetchStocks() {
    this.stockService.GetAssest().subscribe({
      next: (res: any) => {
        this.assets = res;
        if(typeof res == typeof []){
          this.assets.reverse();
        }
        console.log(res);

        res?.forEach((data:any) => {
          console.log(data);

            if(data?.assetType == "STOCK"){
              this.allInvestedPrice += data?.assetPrice * data?.assetQuantity
              this.stockInvestedPrice += data?.assetPrice * data?.assetQuantity
            }else if(data?.assetType == "CRYPTO"){
              this.allInvestedPrice += data?.assetPrice * data?.assetQuantity
              console.log(this.allInvestedPrice +"=-098765434567890-")
              this.cryptoInvestedPrice += data?.assetPrice * data?.assetQuantity;
              this.cryptoQuantityMap.set(data?.assetName?.toLowerCase(),data.assetQuantity);
            }
        });

        const temp:any[] = []
        res?.forEach((d:any)=>{
          temp?.push(d?.assetSymbol?.toLowerCase());
        })

        this.symbols = temp;
      },
    });
  }

  ngOnInit() {
    this.items = [
      { label: 'ALL', icon: 'pi pi-fw pi-calendar' },
      { label: 'Stocks', icon: 'pi pi-fw pi-shield' },
      { label: 'Crypto', icon: 'pi pi-fw pi-bitcoin' },
    ];
  }
}
