import { Component, OnInit, ViewChild } from '@angular/core';
import { ActivatedRoute, Router } from '@angular/router';
import { MenuItem, MessageService } from 'primeng/api';
import { AddCommentModalComponent } from 'src/app/shared/components/add-comment-modal/add-comment-modal.component';
import { UpdateLeadStatusModalComponent } from 'src/app/shared/components/update-lead-status-modal/update-lead-status-modal.component';
import { IInvestorListResponse, Investor } from 'src/models/investor';
import { CrmService } from 'src/services/crm.service';
import { InvestorService } from 'src/services/investor.service';

@Component({
  selector: 'app-investor-list',
  templateUrl: './investor-list.component.html',
  styleUrls: ['./investor-list.component.scss'],
})
export class InvestorListComponent implements OnInit {
  investorList: Array<Investor> = [];
  loading: boolean = true;
  items: MenuItem[] = [];
  selectedInvestorID: number = 0;
  offset: number = 0;
  limit: number = 10;
  totalCount: number = 0;
  searchTerm: string = '';
  editingLeadStatusIvestorId: number | null;

  @ViewChild(UpdateLeadStatusModalComponent)
  updateLeadStatusModalComponent: UpdateLeadStatusModalComponent =
    new UpdateLeadStatusModalComponent();
  @ViewChild(AddCommentModalComponent)
  addCommentModalComponent: AddCommentModalComponent =
    new AddCommentModalComponent();

  constructor(
    private router: Router,
    private investorService: InvestorService,
    public messageService: MessageService,
    private crmService: CrmService,
    private route: ActivatedRoute
  ) {}

  ngOnInit(): void {
    this.items = [
      { label: 'View', icon: 'pi pi-eye', command: () => this.navigate() },
      { label: 'Delete', icon: 'pi pi-trash' },
    ];
    this.route.queryParams.subscribe((params) => {
      this.offset = params['offset'] ? +params['offset'] : 0;
      this.limit = params['limit'] ? +params['limit'] : 10;

      this.getInvestorList();
    });
    // Fetch the initial data using the offset and limit
  }

  getInvestorList() {
    this.loading = true;
    this.investorService
      .fetchInvestorList(this.limit, this.offset, this.searchTerm)
      .subscribe({
        next: (res: IInvestorListResponse) => {
          this.totalCount = res.data.count;
          this.investorList = res.data.users;
          console.log('Investor List Response:', res);
        },
        error: (err) => {
          this.messageService.add({
            severity: 'error',
            summary: 'Something went wrong',
            detail: err.error.message,
          });
          console.error('Investor List Error:', err);
        },
        complete: () => {
          this.loading = false;
        },
      });
  }

  navigate() {
    this.router.navigate(['investor-details', this.selectedInvestorID]);
  }

  paginate(event: any) {
    // Update the offset and limit based on the selected page
    this.offset = event.first;
    this.limit = event.rows;

    // Update the URL with the new offset and limit
    this.router.navigate([], {
      relativeTo: this.route,
      queryParams: { offset: this.offset, limit: this.limit },
      queryParamsHandling: 'merge',
    });
  }

  onSearchInvestor() {
    this.offset = 0;
    this.getInvestorList();
  }

  openLeadStatusPopUp(investor: Investor) {
    this.updateLeadStatusModalComponent.showDialog(
      'Update Lead Status',
      'Lead Status',
      investor.id,
      investor.lead_status
    );
  }

  openProfileCompletionStatusPopUp(investor: Investor) {
    this.updateLeadStatusModalComponent.showDialog(
      'Update Profile Completion Status',
      'Profile Completion Status',
      investor.id,
      investor.profile_completion_status
    );
  }

  openAddCommentPopUp(investor: Investor) {
    this.addCommentModalComponent.showDialog(investor.id);
  }

  addCommentOnSubmit(event: any) {
    console.log('addCommentOnSubmit', event);
    this.crmService.addComment(event['user_id'], event['comment']).subscribe(
      (res) => {
        console.log(res);
        for (let i = 0; i < this.investorList.length; i++) {
          if (this.investorList[i].id === event['user_id']) {
            this.investorList[i].last_comment = event['comment'];
          }
        }
      },
      (err) => {
        console.error(err);
      }
    );
  }

  updateLeadStatusOnSubmit(event: any) {
    console.log('updateLeadStatusOnSubmit', event);
    if (event['type'] === 'Update Lead Status') {
      this.crmService
        .updateLeadStatus(event['user_id'], event['lead_status'])
        .subscribe(
          (res) => {
            console.log(res);
            for (let i = 0; i < this.investorList.length; i++) {
              if (this.investorList[i].id === event['user_id']) {
                this.investorList[i].lead_status = event['lead_status'];
              }
            }
          },
          (err) => {
            console.error(err);
          }
        );
    } else if (event['type'] === 'Update Profile Completion Status') {
      this.crmService
        .updateProfileCompletionStatus(event['user_id'], event['lead_status'])
        .subscribe(
          (res) => {
            console.log(res);
            for (let i = 0; i < this.investorList.length; i++) {
              if (this.investorList[i].id === event['user_id']) {
                this.investorList[i].profile_completion_status =
                  event['lead_status'];
              }
            }
          },
          (err) => {
            console.error(err);
          }
        );
    }
  }
}
