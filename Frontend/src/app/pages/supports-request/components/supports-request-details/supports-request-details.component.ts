import {
  IFetchNoteListResponse,
  INote,
  Note,
} from 'src/models/note';
import { Component, OnInit, ViewChild } from '@angular/core';
import { Location } from '@angular/common';
import { AddNoteModalComponent } from 'src/app/shared/components/add-note-modal/add-note-modal.component';
import { ActivatedRoute } from '@angular/router';
import { IStaffMemberResponse, StaffMember } from 'src/models/staff-member';
import { SupportRequestService } from 'src/services/support_request.service';
import { IAddNoteResponse } from 'src/models/note';
import { IAddCommentResponse, Comment, ICommentListResponse, IComment } from 'src/models/comment';
import { StatusFilter } from 'src/constants/global-constants';
import { IGenericResponse } from 'src/models/genericModels';
import { MessageService } from 'primeng/api';
import { ISupportRequestDetailResponse, SupportRequest } from 'src/models/support-request';

@Component({
  selector: 'app-supports-request-details',
  templateUrl: './supports-request-details.component.html',
  styleUrls: ['./supports-request-details.component.scss'],
})
export class SupportsRequestDetailsComponent implements OnInit {
  sr_notes: Array<Note> = [];
  sr_comments: Array<Comment> = [];
  supportRequest: SupportRequest;
  id: string | null = null;

  @ViewChild(AddNoteModalComponent)
  addNoteModal: AddNoteModalComponent = new AddNoteModalComponent();

  dialogTitle: string = '';
  modalType: string = 'note';
  sr_List: Array<StaffMember> = [];
  noteList: Array<INote> = [];
  commentList: Array<IComment> = [];
  selectedStaffMember: StaffMember;
  statusList: Array<string> = StatusFilter.changeSupportRequestOption;
  selectedStatus: string = '';

  constructor(
    private location: Location,
    private route: ActivatedRoute,
    private SupportRequestService: SupportRequestService,
    public messageService: MessageService
  ) { }

  ngOnInit(): void {
    this.route.paramMap.subscribe((params) => {
      this.id = params.get('id');
      this.getStaffMember();
      this.getNotes();
      this.getComments();
    });
  }

  addNote(note: string) {
    var body = {
      support_request_id: this.id,
      note: note,
    };
    this.SupportRequestService.addSupportRequestNote(body).subscribe({
      next: (res: IAddNoteResponse) => {
        console.log('Support Request Add Note Response:', res);
        this.getNotes();
      },
      error: (err) => {
        console.error('Support Request Add Note Error:', err);
      },
    });
  }

  getNotes() {
    if (this.id) {
      this.SupportRequestService.fetchNoteList(this.id).subscribe({
        next: (res: IFetchNoteListResponse) => {
          this.noteList = res.data.notes.reverse();
          console.log('Support Request Note List Response:', res);
        },
        error: (err) => {
          console.error('Support Request Note List Error:', err);
        },
      });
    }
  }

  getComments() {
    if (this.id) {
      this.SupportRequestService.fetchCommentList(this.id).subscribe({
        next: (res: ICommentListResponse) => {
          this.commentList = res.data.comments.reverse();
          console.log('Support Request Comment List Response:', res);
        },
        error: (err) => {
          console.error('Support Request Comment List Error:', err);
        },
      });
    }
  }

  addComment(comment: string) {
    var body = {
      support_request_id: this.id,
      comment: comment,
    };
    this.SupportRequestService.addSupportRequestComment(body).subscribe({
      next: (res: IAddCommentResponse) => {
        console.log('Support Request Add Comment Response:', res);
        this.getComments();
      },
      error: (err) => {
        console.error('Support Request Add Comment Error:', err);
      },
    });
  }

  showNoteModal() {
    this.modalType = 'note';
    this.dialogTitle = 'Add Note';
    this.addNoteModal.showDialog();
  }

  showCommentModal() {
    this.modalType = 'comment';
    this.dialogTitle = 'Add Comment';
    this.addNoteModal.showDialog();
  }

  navigateBack() {
    this.location.back();
  }

  getStaffMember() {
    this.SupportRequestService.fetchStaffMemberList().subscribe({
      next: (res: IStaffMemberResponse) => {
        this.sr_List = res.data.staff;
        this.getSupportRequestDetails();
        console.log('Staff Member List Response:', res);
      },
      error: (err) => {
        console.error('Staff Member List Error:', err);
      },
    });
  }

  updateStatus() {
    var body = {
      id: this.id,
      status: this.selectedStatus,
    };

    this.SupportRequestService.updateSupportRequestStatus(body).subscribe({
      next: (res: IGenericResponse) => {
        this.messageService.add({ severity: 'success', summary: res.message });
        this.getComments();
        console.log(res.message);
      },
      error: (err) => {
        this.messageService.add({
          severity: 'error',
          summary: err.error.message,
        });
        console.error(err.error.message);
      },
    });
  }

  updateAssignee() {
    var body = {
      id: this.id,
      assigned_to_id: this.selectedStaffMember.id,
    };

    this.SupportRequestService.updateSupportRequestAssignee(body).subscribe({
      next: (res: IGenericResponse) => {
        this.messageService.add({ severity: 'success', summary: res.message });
        this.getComments();
        console.log(res.message);
      },
      error: (err) => {
        this.messageService.add({
          severity: 'error',
          summary: err.error.message,
        });
        console.error(err.error.message);
      },
    });
  }

  getSupportRequestDetails() {
    if(this.id) {
      this.SupportRequestService.fetchSupportRequestDetail(this.id)
      .subscribe({
        next: (res: ISupportRequestDetailResponse) => {
          this.supportRequest = res.data;
          this.sr_List.forEach(staff => {
            if(staff.id == res.data.assigned_to) {
              this.selectedStaffMember = staff;
            }
          });
          this.selectedStatus = this.supportRequest.status;
          console.log("Support Request Detail:", res);
        },
        error: (err) => {
          console.error("Support Request Detail Error:", err);
        }
      });
    }
  }
}
