import { Component, OnInit } from '@angular/core';
import { SelectItem } from 'primeng/api';
import { Stock } from 'src/models/stock';
import { StockService } from 'src/services/stock.service';

@Component({
  selector: 'app-test',
  templateUrl: './test.component.html',
  styleUrls: ['./test.component.scss'],
})
export class TestComponent implements OnInit {
  filteredStocks: Stock[] = [];
  sortOptions: SelectItem[];
  stocks: Stock[] = [];

  ngOnInit(): void {
    this.fetchStocks();
  }
  constructor(private stockService: StockService) {
    this.sortOptions = [
      { label: 'Symbol', value: 'symbol' },
      { label: 'Company Name', value: 'name' },
      { label: 'Price', value: 'price' },
      { label: 'Change (%)', value: 'changePercentage' },
    ];
  }
  fetchStocks(): void {
    this.stockService.getStocks().subscribe((data) => {
      this.stocks = data;
      this.filteredStocks = data;
    });
  }
}
