import { Component, OnInit } from '@angular/core';
import { ActivatedRoute, Router } from '@angular/router';
import { MessageService } from 'primeng/api';
import { InvestableAsset, IInvestableAssetResponse } from 'src/models/asset';
import { InvestBody, InvestMultippleResponse } from 'src/models/investor';
import { AssetService } from 'src/services/asset.service';
import { InvestorService } from 'src/services/investor.service';

@Component({
  selector: 'app-investable-asset-list',
  templateUrl: './investable-asset-list.component.html',
  styleUrls: ['./investable-asset-list.component.scss'],
})
export class InvestableAssetListComponent implements OnInit {
  investmentId: string = '';
  investableAssetsList: Array<InvestableAsset> = [];
  invest: Array<InvestBody> = [];
  loading: boolean = false;

  constructor(
    private assetService: AssetService,
    public messageService: MessageService,
    private route: ActivatedRoute,
    private router: Router,
    private investmentService: InvestorService
  ) {}

  ngOnInit(): void {
    this.route.paramMap.subscribe((params) => {
      this.investmentId = params.get('id') ?? '';
      this.getInvestableAssets();
    });
  }

  getInvestableAssets() {
    if (this.investmentId) {
      this.assetService.fetchInvestableAsset(this.investmentId).subscribe({
        next: (res: IInvestableAssetResponse) => {
          this.investableAssetsList = res.data.investable_assets;
          this.investableAssetsList.forEach((asset) => {
            this.invest.push({
              investment_id: Number(this.investmentId),
              asset_id: asset.id,
              amount: 0,
            });
          });
          console.log('Investable Assets List Response:', res);
        },
        error: (err) => {
          this.displayToast('error', 'Something went wrong', err.error.message);
          console.error('Investable Assets List Error:', err);
        },
      });
    }
  }

  submitInvestment() {
    this.loading = true;
    this.invest.forEach((asset) => (asset.error = ''));
    let body = this.invest.filter((invest) => invest.amount != 0);
    this.investmentService.investFDAmountInAssetMultiple(body).subscribe({
      next: (res: InvestMultippleResponse) => {
        console.log(res.message, res.data);
        this.displayToast('success', 'Success', res.message);
        this.loading = false;
      },
      error: (err) => {
        this.displayToast('error', 'Something went wrong', err.error.message);
        let data: Array<InvestBody> = err.error.data;
        data.forEach((asset) => {
          this.invest.forEach((item) => {
            if (item.asset_id == asset.asset_id) {
              item.error = asset.error;
              item.status = asset.status;
            }
          });
        });
        console.error(err.error.message + ' :', err.error);
        this.loading = false;
      },
    });
  }

  getTotalInvestmentAmount(): number {
    var total: number = 0;
    this.invest.forEach((asset) => (total = total + asset.amount));
    return total;
  }

  getInvestmentCount(): number {
    var count: number = 0;
    this.invest.forEach((asset) => {
      if (asset.amount) {
        count = count + 1;
      }
    });
    return count;
  }

  displayToast(severity: string, summary: string, detail: string) {
    this.messageService.add({
      severity: severity,
      summary: summary,
      detail: detail,
    });
  }
  openInvestmentInNewTab(id: string) {
    const url = this.router.serializeUrl(
      this.router.createUrlTree(['/assets', 'list', id])
    );
    window.open(url, '_blank');
  }
}
