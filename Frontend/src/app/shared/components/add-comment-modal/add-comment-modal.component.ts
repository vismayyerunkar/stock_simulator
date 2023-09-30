import { Component, EventEmitter, Input, OnInit, Output } from '@angular/core';

@Component({
  selector: 'app-add-comment-modal',
  templateUrl: './add-comment-modal.component.html',
  styleUrls: ['./add-comment-modal.component.scss'],
})
export class AddCommentModalComponent implements OnInit {
  display: boolean = false;
  comment: string = '';
  @Output() submit = new EventEmitter();
  user_id: number | null;

  constructor() {}

  ngOnInit(): void {}

  onSubmit() {
    if (this.comment != null && this.comment.length > 0) {
      this.submit.emit({
        user_id: this.user_id,
        comment: this.comment,
      });
    }
    this.hideDialog();
  }

  showDialog(user_id: number) {
    this.user_id = user_id;
    this.display = true;
  }

  hideDialog() {
    this.display = false;
    this.comment = '';
    this.user_id = null;
  }
}
