import { Component, OnInit } from '@angular/core';

@Component({
  selector: 'app-wallet-pie-chart',
  templateUrl: './wallet-pie-chart.component.html',
  styleUrls: ['./wallet-pie-chart.component.scss'],
})
export class WalletPieChartComponent implements OnInit {
  constructor() {}
  data: any;

  options: any;

  ngOnInit() {
    const documentStyle = getComputedStyle(document.documentElement);
    const textColor = documentStyle.getPropertyValue('--text-color');

    this.data = {
      labels: ['Equity', 'Crypto', 'NFT', 'Funds'],
      datasets: [
        {
          data: [150, 70, 100, 120],
          backgroundColor: [
            documentStyle.getPropertyValue('--blue-500'),
            documentStyle.getPropertyValue('--yellow-500'),
            documentStyle.getPropertyValue('--green-500'),
            documentStyle.getPropertyValue('--red-500'),
          ],
          hoverBackgroundColor: [
            documentStyle.getPropertyValue('--blue-400'),
            documentStyle.getPropertyValue('--yellow-400'),
            documentStyle.getPropertyValue('--green-400'),
            documentStyle.getPropertyValue('--red-400'),
          ],
        },
      ],
    };

    this.options = {
      cutout: '60%',
      plugins: {
        legend: {
          labels: {
            color: textColor,
          },
        },
      },
    };
  }
}
