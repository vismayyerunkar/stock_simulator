import { Component, Input, OnInit } from '@angular/core';
import { ActivatedRoute } from '@angular/router';
import { InvestmentSummary, InvestmentSummaryResponse, InvestorDetail } from 'src/models/investor';
import { InvestorService } from 'src/services/investor.service';

@Component({
  selector: 'app-user-details',
  templateUrl: './user-details.component.html',
  styleUrls: ['./user-details.component.scss']
})
export class UserDetailsComponent implements OnInit {

  @Input()
  investorDetail!: InvestorDetail;
  userid: string | null = null;
  investmentSummary: InvestmentSummary;

  constructor(private investorService: InvestorService,
    private route: ActivatedRoute) { 
    this.route.paramMap.subscribe((params) => {
      this.userid = params.get('id');
      if (this.userid) {
        this.getInvestorSummary();
      }
    });
  }

  ngOnInit(): void {
  }

  getInvestorSummary() {
    if(this.userid) {
      this.investorService.fetchInvestmentSummary(this.userid)
      .subscribe({
        next: (res: InvestmentSummaryResponse) => {
          this.investmentSummary = res.data;
          console.log("Investment Summary:", res);
        },
        error: (err) => console.error("Investment Summary Error", err)
      });
    }
  }

  refactorKey(key: string) {
    return key.replace(/_/g, ' ');
  }

}
