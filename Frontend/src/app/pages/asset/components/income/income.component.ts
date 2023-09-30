import { Component, OnInit } from '@angular/core';
import { Income } from 'src/app/models/income';

@Component({
  selector: 'app-income',
  templateUrl: './income.component.html',
  styleUrls: ['./income.component.scss'],
})
export class IncomeComponent implements OnInit {
  loading: boolean = true;
  incomeList: Array<Income> = [];

  constructor() {}

  ngOnInit() {
    this.getInvestments();
  }

  getInvestments() {
    this.loading = true;
    for (let i = 1; i <= 100; i++) {
      var income: Income = {
        id: i,
        amount: 40000,
        purpose: 'Sales',
        transaction_dtm: 'sample',
        consumed_on: new Date()
      };
      this.incomeList.push(income);
    }
    this.loading = false;
  }
}
