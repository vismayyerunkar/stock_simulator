import { ActivatedRoute } from '@angular/router';
import { Component, OnInit } from '@angular/core';
import { Location } from '@angular/common';
import {
  InvestmentDetailsResponse,
  InvestmentDetails,
} from 'src/models/investor';
import { InvestorService } from 'src/services/investor.service';

@Component({
  selector: 'app-fd-investment-detail',
  templateUrl: './fd-investment-detail.component.html',
  styleUrls: ['./fd-investment-detail.component.scss'],
})
export class FdInvestmentDetailComponent implements OnInit {
  id: string;
  investorDetails!: InvestmentDetails;

  constructor(
    private route: ActivatedRoute,
    private location: Location,
    private investorService: InvestorService
  ) {}

  ngOnInit(): void {
    this.route.paramMap.subscribe((params) => {
      this.id = params.get('id') ?? '';
      this.getInvestorDetails();
    });
  }
  
  getInvestorDetails() {
    this.investorService.fetchInvestmentDetails(this.id).subscribe({
      next: (res: InvestmentDetailsResponse) => {
        this.investorDetails = res.data;
        console.log('Investment Details Response:', res);
      },
      error: (err: any) => {
        console.error('Investment Details Error:', err);
      },
    });
  }
  
  navigateBack() {
    this.location.back();
  }
}
