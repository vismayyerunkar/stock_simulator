import { Component, EventEmitter, Input, OnInit, Output } from '@angular/core';

@Component({
  selector: 'app-add-note-modal',
  templateUrl: './add-note-modal.component.html',
  styleUrls: ['./add-note-modal.component.scss'],
})
export class AddNoteModalComponent implements OnInit {
  display: boolean = false;
  value: string = '';
  @Input() type: string = '';
  @Output() note = new EventEmitter();
  @Output() comment = new EventEmitter();
  @Input() title = '';

  constructor() {}

  ngOnInit(): void {}

  submit() {
    if (this.type == 'note') {
      this.note.emit(this.value);
    } else if (this.type == 'comment') {
      this.comment.emit(this.value);
    }
    this.hideDialog();
  }

  showDialog() {
    this.display = true;
  }

  hideDialog() {
    this.display = false;
    this.value = '';
  }
}
