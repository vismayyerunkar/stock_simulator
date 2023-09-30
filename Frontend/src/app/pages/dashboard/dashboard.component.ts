import { Component, OnInit } from '@angular/core';
import { SocketService } from '../../../services/socketService';  
import { Routes } from '@angular/router';
import { StockDetailsComponent } from '../stock-details/stock-details.component';

@Component({
  selector: 'app-dashboard',
  templateUrl: './dashboard.component.html',
  styleUrls: ['./dashboard.component.scss'],
})

export class DashboardComponent implements OnInit {

  public topLoosers:any[] = [];
  public topGainers:any[] = [];
  public topLosersCryptos:any[] = [];
  public mostTradedCryptos:any[] = [];

  routes: Routes = [
    // ... other routes ...

    // Define the stock-details route
    { path: 'stock-details/:1', component: StockDetailsComponent },
  ];

  constructor(private socketService: SocketService){

      this.topLoosers = [...new Array(6)].map(() => 0);
      this.topGainers = [...new Array(6)].map(() => 0);

      socketService.fetchTopStocks()?.subscribe((data: any) => {
        // this.topLosersStocks= data;
        this.topGainers = data?.gainers;
        this.topLoosers = data?.losers;
        this.cleanUp();
      });

      socketService.fetchTopCryptos()?.subscribe((data: any) => {
        // this.topLosersStocks= data;
        this.mostTradedCryptos = data?.gainers;
        this.topLosersCryptos = data?.loosers;
      })
  }


  cleanUp(){
    this.topGainers = this.topGainers.filter((stock)=>stock?.meta && stock?.symbol != "NIFTY 50");
    this.topLoosers = this.topLoosers.filter((stock)=>stock?.meta && stock?.symbol != "NIFTY 50");
    this.topGainers.length = 6;
    this.topLoosers.length = 6;
  }

  stocksData: any[] = [];
  


  // topLosersCryptos = [
  //   {
  //     title: 'Crypto 3',
  //     value: '₹119.10',
  //     percentageChange: '-8%',
  //     isNew: false,
  //   },
  //   {
  //     title: 'Crypto 4',
  //     value: '₹119.10',
  //     percentageChange: '-12%',
  //     isNew: false,
  //   },
  //   {
  //     title: 'Crypto 4',
  //     value: '₹119.10',
  //     percentageChange: '-12%',
  //     isNew: false,
  //   },
  //   {
  //     title: 'Crypto 4',
  //     value: '₹119.10',
  //     percentageChange: '-12%',
  //     isNew: false,
  //   },
  //   // Add more top losers cryptos data here
  // ];
  newIPOs = [
    { name: 'IPO 1', image: 'url_to_image_1' },
    { name: 'IPO 2', image: 'url_to_image_2' },
    // ... add more IPO data ...
  ];
  ngOnInit(): void {}
}