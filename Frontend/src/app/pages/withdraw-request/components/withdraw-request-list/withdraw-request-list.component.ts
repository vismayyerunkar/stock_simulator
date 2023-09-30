import {
  IWithdrawListResponse,
  WithdrawRequest,
} from './../../../../../models/withdraw';
import { Component, OnInit } from '@angular/core';
import { StatusFilter } from 'src/constants/global-constants';
import { WithdrawRequestService } from 'src/services/withdraw.service';
import { MessageService } from 'primeng/api';

@Component({
  selector: 'app-withdraw-request-list',
  templateUrl: './withdraw-request-list.component.html',
  styleUrls: ['./withdraw-request-list.component.scss'],
})
export class WithdrawRequestListComponent implements OnInit {
  offset: number = 0;
  limit: number = 10;
  totalCount: number = 0;
  loading: boolean = true;
  statusOption: Array<string> = StatusFilter.withdrawListOptions;
  selectedStatus: Array<String> = [];
  withdrawList: Array<WithdrawRequest> = [];

  constructor(
    private withdrawRequestService: WithdrawRequestService,
    public messageService: MessageService
  ) {}

  ngOnInit(): void {
    this.getWithdraws();
  }
  getWithdraws() {
    this.loading = true;
    this.withdrawRequestService
      .fetchWithdrawList(this.offset, this.limit, this.selectedStatus.toString())
      .subscribe({
        next: (res: IWithdrawListResponse) => {
          this.totalCount = res.data.count;
          this.withdrawList = res.data.withdraw_requests;
          console.log('Withdraw Request List Response:', res);
        },
        error: (err) => {
          this.messageService.add({
            severity: 'error',
            summary: 'Something went wrong',
            detail: err.error.message,
          });
          console.error('Withdraw Request List Error:', err);
        },
        complete: () => {
          this.loading = false;
        },
      });
  }
  filterChange() {
    this.offset = 0;
    this.getWithdraws();
  }
  clearFilter() {
    this.selectedStatus = [];
    this.offset = 0;
    this.limit = 10;
    this.getWithdraws();
  }
  paginate(event: any) {
    this.offset = event.first;
    this.limit = event.rows;
    this.getWithdraws();
  }
}
