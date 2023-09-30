import { PrimeNgModule } from './prime-ng.module';
import { UserDetailsComponent } from './../components/user-details/user-details.component';
import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule, ReactiveFormsModule } from '@angular/forms';
import { AddNoteModalComponent } from './../components/add-note-modal/add-note-modal.component';
import { UpdateLeadStatusModalComponent } from '../components/update-lead-status-modal/update-lead-status-modal.component';
import { AddCommentModalComponent } from '../components/add-comment-modal/add-comment-modal.component';

@NgModule({
  declarations: [
    UserDetailsComponent,
    AddNoteModalComponent,
    UpdateLeadStatusModalComponent,
    AddCommentModalComponent,
  ],
  imports: [CommonModule, PrimeNgModule, FormsModule, ReactiveFormsModule],
  exports: [
    UserDetailsComponent,
    AddNoteModalComponent,
    UpdateLeadStatusModalComponent,
    AddCommentModalComponent,
    FormsModule,
    ReactiveFormsModule,
  ],
})
export class SharedModule {}
