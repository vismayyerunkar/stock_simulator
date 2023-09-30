import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable, of } from 'rxjs';
import { ITransactionsResponse } from 'src/models/Transactions';
import { Stock } from 'src/models/stock';
import { IWatchListResponse } from 'src/models/watchlist';
import { environment } from 'src/environments/environment';


@Injectable({
  providedIn: 'root',
})
export class StockService {
  constructor(private http: HttpClient) {}

  getHistoricalPrices(symbol: string): any {
    throw new Error('Method not implemented.');
  }
  private stocks: Stock[] = [
    {
      symbol: 'TCS.NS',
      name: 'TCS Ltd',
      price: 79.899,
      changePercentage: 34.89,
    },
    {
      symbol: 'AAPL',
      name: 'Apple Inc.',
      price: 150.55,
      changePercentage: -1.23,
    },
    {
      symbol: 'AAPL',
      name: 'Apple Inc.',
      price: 150.55,
      changePercentage: -1.23,
    },
    {
      symbol: 'AAPL',
      name: 'Apple Inc.',
      price: 150.55,
      changePercentage: -1.23,
    },
    {
      symbol: 'AAPL',
      name: 'Apple Inc.',
      price: 150.55,
      changePercentage: -1.23,
    },
    {
      symbol: 'AAPL',
      name: 'Apple Inc.',
      price: 150.55,
      changePercentage: -1.23,
    },
    {
      symbol: 'AAPL',
      name: 'Apple Inc.',
      price: 150.55,
      changePercentage: -1.23,
    },
    {
      symbol: 'AAPL',
      name: 'Apple Inc.',
      price: 150.55,
      changePercentage: -1.23,
    },
    {
      symbol: 'AAPL',
      name: 'Apple Inc.',
      price: 150.55,
      changePercentage: -1.23,
    },
    {
      symbol: 'AAPL',
      name: 'Apple Inc.',
      price: 150.55,
      changePercentage: -1.23,
    },
    {
      symbol: 'AAPL',
      name: 'Apple Inc.',
      price: 150.55,
      changePercentage: -1.23,
    },

    // Add more mock data as needed
  ];

  getStocks(): Observable<Stock[]> {
    return of(this.stocks);
  }
  
  fetchWatchList(): Observable<IWatchListResponse> {
    const token = `Bearer ${localStorage.getItem('authToken')}`;
    const url = `${environment.baseUrl}/api/watchlist/watchlists`;
    let httpOptions = {
      headers: new HttpHeaders({ Authorization: token }),
    };
    return this.http.get<IWatchListResponse>(url, httpOptions);
  }

  apiKey = 'GBXGRVSC44LFEGD8'

  searchStockSymbols(keyword: string): Observable<any> {
    const url = `https://www.alphavantage.co/query?function=SYMBOL_SEARCH&keywords=${keyword}&apikey=${this.apiKey}`;
    return this.http.get(url);
  }

  fetchTransactions(): Observable<ITransactionsResponse> {
    const token = `Bearer ${localStorage.getItem('authToken')}`;
    const url = `${environment.baseUrl}/api/transactions/`;
    let httpOptions = {
      headers: new HttpHeaders({ Authorization: token }),
    };
    return this.http.get<ITransactionsResponse>(url, httpOptions);
  }
}
