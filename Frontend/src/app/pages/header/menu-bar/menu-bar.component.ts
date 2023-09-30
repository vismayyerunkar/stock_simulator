import { SocketService } from 'src/services/socketService';
import { StockService } from 'src/services/stock.service';
import { Category } from 'src/models/feedback';
import { environment } from 'src/environments/environment';
import axios from 'axios';
import { Component, OnInit } from '@angular/core';
import { Router } from '@angular/router';
import { MenuItem } from 'primeng/api';
import { UserAuthService } from 'src/services/user-auth.service';

@Component({
  selector: 'app-menu-bar',
  templateUrl: './menu-bar.component.html',
  styleUrls: ['./menu-bar.component.scss'],
})
export class MenuBarComponent implements OnInit {
  items: MenuItem[] = [];
  visibleSidebar: any;
  userData: any;
  funds: string | number;
  router: Router;
  searchTerm = '';
  stockSymbols: any[] = [];
  cryptoMap = new Map();

  // randomProfilePic:string

  constructor(
    private userAuthService: UserAuthService,
    private stockService: StockService
  ) {
    // const temp:any = localStorage.getItem("loggedUserRandomProfile")

    // if(typeof temp == typeof undefined){
    //   localStorage.setItem("loggedUserRandomProfile",`https://randomuser.me/api/portraits/men/${Math.floor(Math.random()*10)}.jpg`)
    // }else{
    //   this.randomProfilePic = JSON.parse(temp == "undefined" ? "https://randomuser.me/api/portraits/men/12.jpg" : temp);
    // }

    this.userData = JSON.parse(localStorage.getItem('user') ?? '{}');
    axios.interceptors.request.use(function (config) {
      config.headers.Authorization = `Bearer ${localStorage.getItem(
        'authToken'
      )}`;
      return config;
    });
    axios
      .get(`${environment.baseUrl}/api/user/loggedUser`)
      .then((res: any) => {
        console.log(res.data);
        this.funds = res.data?.user?.availableTokens;
      })
      .catch((err: any) => {
        console.log('Error occured : ', err);
      });
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

  searchAsset(searchValue?: any) {
    console.log(searchValue ?? this.searchTerm);
    this.createCryptoMap()
      .then(() => {
        console.log(
          this.cryptoMap.get(
            searchValue?.symbol?.split('.')[0] ?? this.searchTerm
          )
        );
        window.location.href = `/stock-details;title=${
          searchValue?.symbol?.split('.')[0] ?? this.searchTerm
        };type=${
          this.cryptoMap.get(
            searchValue?.symbol?.split('.')[0] ?? this.searchTerm
          )
            ? 'CRYPTO'
            : 'STOCK'
        }`;
      })
      .catch((err) => {
        console.log(err);
        alert('An error occured please try again');
      });
  }

  ngOnInit() {
    this.items = [
      {
        label: 'Assests',
        icon: 'pi pi-fw pi-server',
        items: [
          {
            label: 'Stocks',
            icon: 'pi pi-fw pi-shield',
            routerLink: ['/stocks'],
          },
          {
            label: 'Crypto',
            icon: 'pi pi-fw pi-bitcoin',
            routerLink: ['/crypto'],
          },
          {
            separator: true,
          },
          {
            label: 'New',
            icon: 'pi pi-fw pi-external-link',
            routerLink: ['/dashboard'],
          },
        ],
      },
      {
        label: 'Goal',
        icon: 'pi pi-fw pi-map',
        routerLink: ['/goal'],
      },
      {
        label: 'Portfolio',
        icon: 'pi pi-fw pi-server',
        routerLink: ['/portfolio'],
      },
      {
        label: 'Watchlist',
        icon: 'pi pi-fw pi-sliders-h',
        routerLink: ['/watchlist'],
      },
      {
        label: 'AI-GPT',
        icon: 'pi pi-search-plus',
        routerLink: ['/gpt'],
      },
    ];
  }

  onInputChange() {
    console.log(this.searchTerm);
    if (this.searchTerm.trim() !== '') {
      this.stockService
        .searchStockSymbols(this.searchTerm)
        .subscribe((data: any) => {
          console.log(data.bestMatches);
          this.stockSymbols = data.bestMatches;
        });
    } else {
      this.stockSymbols = [];
    }
  }

  logout() {
    localStorage.removeItem('authToken');
    localStorage.removeItem('user');
    window.location.href = '/login';
  }
}
