import { Component, OnInit } from '@angular/core';
import { ActivatedRoute, Router } from '@angular/router';
import { MessageService } from 'primeng/api';
import {
  InvestmentUtilization,
  IInvestmentUtilizationResponse,
} from 'src/models/asset';
import { AssetService } from 'src/services/asset.service';
import { InvestorService } from 'src/services/investor.service';

@Component({
  selector: 'app-fd-investment-utilization',
  templateUrl: './fd-investment-utilization.component.html',
  styleUrls: ['./fd-investment-utilization.component.scss'],
})
export class FdInvestmentUtilizationComponent implements OnInit {

  investmentId: string = '';
  investmentUtilizationList: Array<InvestmentUtilization> = [];

  constructor(
    private assetService: AssetService,
    private investmentService: InvestorService,
    public messageService: MessageService,
    private router: Router,
    private route: ActivatedRoute
  ) {}

  ngOnInit(): void {
    this.route.paramMap.subscribe((params) => {
      this.investmentId = params.get('id') ?? '';
      this.getInvestmentUtilization();
    });
  }
  getInvestmentUtilization() {
    if(this.investmentId) {
      this.investmentService.fetchInvestmentUtilization(this.investmentId).subscribe({
        next: (res: IInvestmentUtilizationResponse) => {
          this.investmentUtilizationList = res.data.results;
          console.log('Investment Utilization List Response:', res);
        },
        error: (err) => {
          this.messageService.add({
            severity: 'error',
            summary: 'Something went wrong',
            detail: err.error.message,
          });
          console.error('Investment Utilization List Error:', err);
        }
      });
    }
  }

  openInvestmentInNewTab(assetId: string) {
    const url = this.router.serializeUrl(
      this.router.createUrlTree(['/assets', 'list', assetId])
    );
    window.open(url, '_blank');
  }
}
