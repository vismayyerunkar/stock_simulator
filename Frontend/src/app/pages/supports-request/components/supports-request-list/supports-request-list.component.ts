import { DatePipe } from '@angular/common';
import { Component, OnInit } from '@angular/core';
import { MessageService } from 'primeng/api';
import { StatusFilter } from 'src/constants/global-constants';
import { Category, IFeedbackListCategoryResponse } from 'src/models/feedback';
import { IStaffMemberResponse, StaffMember } from 'src/models/staff-member';
import {
  ISupportRequestListResponse,
  SupportRequest,
} from 'src/models/support-request';
import { FeedbackService } from 'src/services/feedback.service';
import { SupportRequestService } from 'src/services/support_request.service';

@Component({
  selector: 'app-supports-request-list',
  templateUrl: './supports-request-list.component.html',
  styleUrls: ['./supports-request-list.component.scss'],
})
export class SupportsRequestListComponent implements OnInit {
  loading: boolean = true;
  supportRequestList: Array<SupportRequest> = [];
  sr_List: Array<StaffMember> = [];
  selectedStaffMember: StaffMember | null = null;
  statusOption: Array<string> = StatusFilter.supportRequestStatus;
  selectedStatus: string | null = null;
  offset: number = 0;
  limit: number = 10;
  totalCount: number = 0;
  selectedDate: any = null;
  categoryOption: Array<Category> = [];
  selectedCategory: Category | null = null;

  selectedSort: any = null;
  sortOption: any[] = [
    { name: 'ASC Created_on', value: 'created_on' },
    { name: 'DESC Created_on', value: '-created_on' },
  ];

  constructor(
    private supportRequestService: SupportRequestService,
    public messageService: MessageService,
    public datepipe: DatePipe,
    private feedbackService: FeedbackService
  ) {}

  ngOnInit() {
    this.selectedSort = this.sortOption[0];
    this.getCategoryList();
    this.getStaffMember();
    this.getSupportRequestList();
  }

  getSupportRequestList() {
    this.loading = true;
    let status: string = this.selectedStatus ? this.selectedStatus : '';
    let assignedTo: string = this.selectedStaffMember
      ? this.selectedStaffMember.id.toString()
      : '';
    let categoryID: string = this.selectedCategory
      ? String(this.selectedCategory.id)
      : '';

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

    this.supportRequestService
      .fetchSupportRequestList(
        this.offset,
        this.limit,
        status,
        startDate,
        endDate,
        this.selectedSort.value,
        categoryID,
        assignedTo,
        ''
      )
      .subscribe({
        next: (res: ISupportRequestListResponse) => {
          this.totalCount = res.data.count;
          this.supportRequestList = res.data.support_requests;
          console.log('Support Request List Response:', res);
        },
        error: (err) => {
          this.messageService.add({
            severity: 'error',
            summary: 'Something went wrong',
            detail: err.error.message,
          });
          console.error('Support Request List Error:', err);
        },
        complete: () => {
          this.loading = false;
        },
      });
  }

  getCategoryList() {
    this.feedbackService.fetchFeedbackCategoryList().subscribe({
      next: (res: IFeedbackListCategoryResponse) => {
        this.categoryOption = res.data;
        console.log('Category List:', res);
      },
      error: (err) => {
        console.error('Category List Error:', err);
      },
    });
  }

  getStaffMember() {
    this.supportRequestService.fetchStaffMemberList().subscribe({
      next: (res: IStaffMemberResponse) => {
        this.sr_List = res.data.staff;
        console.log('Staff Member List Response:', res);
      },
      error: (err) => {
        console.error('Staff Member List Error:', err);
      },
    });
  }

  changeFilterDate() {
    if (this.selectedDate[0] != null && this.selectedDate[1] != null) {
      this.offset = 0;
      this.getSupportRequestList();
    }
  }

  filterChange() {
    this.offset = 0;
    this.getSupportRequestList();
  }

  clearFilter() {
    this.selectedStatus = null;
    this.selectedStaffMember = null;
    this.selectedCategory = null;
    this.selectedSort = this.sortOption[0];
    this.offset = 0;
    this.limit = 10;
    this.selectedDate = null;
    this.getSupportRequestList();
  }

  paginate(event: any) {
    this.offset = event.first;
    this.limit = event.rows;
    this.getSupportRequestList();
  }
}
