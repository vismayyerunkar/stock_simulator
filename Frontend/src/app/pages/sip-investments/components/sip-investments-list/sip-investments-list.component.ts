import { DatePipe } from '@angular/common';
import { Component, OnInit } from '@angular/core';
import { Router } from '@angular/router';
import { MessageService } from 'primeng/api';
import { StatusFilter } from 'src/constants/global-constants';
import {
  ISIPInvestmentsListResponse,
  SmartInvestment,
} from 'src/models/investor';
import { InvestorService } from 'src/services/investor.service';

@Component({
  selector: 'app-sip-investments-list',
  templateUrl: './sip-investments-list.component.html',
  styleUrls: ['./sip-investments-list.component.scss'],
})
export class SipInvestmentsListComponent implements OnInit {
  sipInvestmentList: Array<SmartInvestment> = [];
  statusOption: Array<string> = StatusFilter.options;
  selectedStatus: string | null = null;
  offset: number = 0;
  limit: number = 10;
  totalCount: number = 0;
  selectedDate: any = null;
  loading: boolean = true;

  selectedSort: any = null;
  sortOption: any[] = [
    { name: 'ASC Created_on', value: 'created_on' },
    { name: 'DESC Created_on', value: '-created_on' },
  ];

  constructor(
    private investorService: InvestorService,
    public messageService: MessageService,
    public datepipe: DatePipe,
    private router: Router
  ) {}

  ngOnInit() {
    this.selectedSort = this.sortOption[0];
    this.getSIPInvestments();
  }

  getSIPInvestments() {
    this.loading = true;
    let status: string = this.selectedStatus ? this.selectedStatus : '';

    let startDate;
    let endDate;
    if (
      this.selectedDate &&
      this.selectedDate[0] != null &&
      this.selectedDate[1] != null
    ) {
      startDate = this.datepipe.transform(this.selectedDate[0], 'yyyy-MM-dd');
      endDate = this.datepipe.transform(this.selectedDate[1], 'yyyy-MM-dd');
    } else {
      startDate = '';
      endDate = '';
    }

    this.investorService
      .fetchSIPInvestmentList(
        this.offset,
        this.limit,
        status,
        startDate,
        endDate,
        this.selectedSort.value,
        ''
      )
      .subscribe({
        next: (res: ISIPInvestmentsListResponse) => {
          this.totalCount = res.data.count;
          this.sipInvestmentList = res.data.investments;
          console.log('SIP Investment List Response:', res);
        },
        error: (err) => {
          this.messageService.add({
            severity: 'error',
            summary: 'Something went wrong',
            detail: err.error.message,
          });
          console.error('SIP Investment List Error:', err);
        },
        complete: () => {
          this.loading = false;
        },
      });
  }

  changeFilterDate() {
    if (this.selectedDate[0] != null && this.selectedDate[1] != null) {
      this.offset = 0;
      this.getSIPInvestments();
    }
  }

  filterChange() {
    this.offset = 0;
    this.getSIPInvestments();
  }

  clearFilter() {
    this.selectedStatus = null;
    this.selectedSort = this.sortOption[0];
    this.offset = 0;
    this.limit = 10;
    this.selectedDate = null;
    this.getSIPInvestments();
  }

  paginate(event: any) {
    this.offset = event.first;
    this.limit = event.rows;
    this.getSIPInvestments();
  }
  
  openInvestor(User: string) {
    const url = this.router.serializeUrl(
      this.router.createUrlTree(['/investor-details', User])
    );
    window.open(url, '_blank');
  }
}
