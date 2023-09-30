import { Component, OnInit } from '@angular/core';
import { MenuItem, MessageService } from 'primeng/api';
import { IFeedbackListResponse, Feedback, IFeedbackListCategoryResponse, Category } from 'src/models/feedback';
import { FeedbackService } from 'src/services/feedback.service';
import { DatePipe } from '@angular/common';
@Component({
  selector: 'app-feedback-list',
  templateUrl: './feedback-list.component.html',
  styleUrls: ['./feedback-list.component.scss'],
})
export class FeedbackListComponent implements OnInit {
  loading: boolean = true;
  feedbackList: Array<Feedback> = [];
  selectedSort: any = null;
  sortOption: any[] = [
    { name: 'ASC Created_on', value: 'created_on' },
    { name: 'DESC Created_on', value: '-created_on' }
  ];
  categoryOption: Array<Category> = [];
  selectedCategory: Category | null = null;
  offset: number = 0;
  limit: number = 5;
  totalCount: number = 0;
  selectedDate: any = null;

  constructor(
    private feedbackService: FeedbackService,
    public messageService: MessageService,
    public datepipe: DatePipe
  ) { }

  ngOnInit() {
    this.selectedSort = this.sortOption[0];
    this.getFeedbackCategories();
    this.getFeedbacks();
  }

  getFeedbackCategories() {
    this.feedbackService.fetchFeedbackCategoryList()
      .subscribe({
        next: (res: IFeedbackListCategoryResponse) => {
          this.categoryOption = res.data;
          console.log("Feedback Category List:", res);
        },
        error: (err) => {
          console.error("Feedback Category List Error:", err);
        }
      });
  }

  getFeedbacks() {
    this.loading = true;
    let categoryID: string = this.selectedCategory ? String(this.selectedCategory.id) : '';

    let startDate;
    let endDate;
    if (this.selectedDate && this.selectedDate[0] != null && this.selectedDate[1] != null) {
      startDate = this.datepipe.transform(this.selectedDate[0], 'yyyy-MM-dd');
      endDate = this.datepipe.transform(this.selectedDate[1], 'yyyy-MM-dd');
    }
    else {
      startDate = '';
      endDate = '';
    }

    this.feedbackService.fetchFeedbackList(this.offset, this.limit, this.selectedSort.value, categoryID, startDate, endDate).subscribe({
      next: (res: IFeedbackListResponse) => {
        this.totalCount = res.data.count;
        this.feedbackList = res.data.feedback_list;
        console.log('Feedback List Response:', res);
      },
      error: (err) => {
        this.messageService.add({ severity: 'error', summary: 'Something went wrong', detail: err.error.message });
        console.error('Feedback List Error:', err);
      },
      complete: () => {
        this.loading = false;
      },
    });
  }

  changeFilterDate() {
    if (this.selectedDate[0] != null && this.selectedDate[1] != null) {
      this.offset = 0;
      this.getFeedbacks();
    }
  }

  filterChange() {
    this.offset = 0;
    this.getFeedbacks();
  }

  clearFilter() {
    this.selectedSort = this.sortOption[0];
    this.selectedCategory = null;
    this.offset = 0;
    this.limit = 5;
    this.selectedDate = null;
    this.getFeedbacks();
  }

  paginate(event: any) {
    this.offset = event.first;
    this.limit = event.rows;
    this.getFeedbacks();
  }
}
