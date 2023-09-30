import { Component, OnInit, Input, OnChanges, SimpleChanges } from '@angular/core';

@Component({
  selector: 'app-crypto',
  templateUrl: './crypto.component.html',
  styleUrls: ['./crypto.component.scss'],
})
export class CryptoComponent implements OnInit, OnChanges {
  @Input() quantityMap: Map<string, number>;
  @Input() currentValue: number;
  @Input() percentChange:number;
  @Input() investedAmt:number;
  data: any;
  options: any;
  names: any[] = [];
  pricePercent: any[] = [];
  tp: any;
  tempBgColorsMap:Map<string,string> = new Map();

  ngOnInit() {
    this.updateData();
  }

  constructor() {
    this.tp = this.quantityMap;
  }

  ngOnChanges(changes: SimpleChanges) {
    this.tp = this.quantityMap;
    if (changes["quantityMap"] && !changes["quantityMap"].firstChange) {
      this.tp = this.quantityMap;
      this.updateData();
    }
  }

  getRandomColor() {
    const letters = "0123456789ABCDEF";
    let color = "#";
    for (let i = 0; i < 6; i++) {
      color += letters[Math.floor(Math.random() * 16)];
    }
    return color;
  }

  updateData() {
    const documentStyle = getComputedStyle(document.documentElement);
    const textColor = documentStyle.getPropertyValue('--text-color');

    this.names = [];
    this.pricePercent = [];
    
    for (let [key, value] of this.tp.entries()) {
      this.names.push(key);
      this.pricePercent.push(value);
      console.log(key,value)
    }

    let tempBgColors = new Array(this.names.length);

    for(let i = 0;i<this.names.length;i++){
      let randomColor = this.getRandomColor();
      while(tempBgColors.includes(randomColor)){
        randomColor = this.getRandomColor();
      }
      tempBgColors[i] = randomColor;
      this.tempBgColorsMap.set(this.names[i],randomColor);
    }


    const backgroundColors = tempBgColors;
    const hoverBackgroundColors = backgroundColors.slice(); // Copy the colors array

    this.data = {
      labels: [...this.names],
      datasets: [
        {
          data: [...this.pricePercent],
          backgroundColor: backgroundColors,
          hoverBackgroundColor: hoverBackgroundColors,
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
