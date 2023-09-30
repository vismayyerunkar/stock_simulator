import { Component, Input, OnInit } from '@angular/core';
import { AssetInvestment, IAssetInvestmentResponse } from 'src/models/asset';
import { AssetService } from 'src/services/asset.service';

@Component({
  selector: 'app-investments',
  templateUrl: './investments.component.html',
  styleUrls: ['./investments.component.scss'],
})
export class InvestmentsComponent implements OnInit {
  loading: boolean = true;
  @Input() assetId: string;
  limit: number = 10;

  investmentList: Array<AssetInvestment> = [];
  constructor(private assetService: AssetService) {}

  ngOnInit() {
    this.getInvestments();
  }

  getInvestments() {
    if(this.assetId) {
      this.assetService.SellAssest().subscribe({
        next: (res: IAssetInvestmentResponse) => {
          this.investmentList = res.data;
          console.log('Asset Investment Response:', res);
        },
        error: (err: any) => {
          console.error('Asset Investment Error:', err);
        },
      });
    }
  }
}
