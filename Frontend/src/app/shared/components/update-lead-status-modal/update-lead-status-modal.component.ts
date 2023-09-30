import { Component, EventEmitter, Input, OnInit, Output } from '@angular/core';

@Component({
  selector: 'app-update-lead-status-modal',
  templateUrl: './update-lead-status-modal.component.html',
  styleUrls: ['./update-lead-status-modal.component.scss'],
})
export class UpdateLeadStatusModalComponent implements OnInit {
  display: boolean = false;
  lead_status: string = '';
  type: string = '';
  label: string = '';
  @Output() submit = new EventEmitter();
  user_id: number | null;

  constructor() {}

  ngOnInit(): void {}

  onSubmit() {
    if (this.lead_status != null && this.lead_status.length > 0) {
      this.submit.emit({
        user_id: this.user_id,
        lead_status: this.lead_status,
        type: this.type,
      });
    }
    this.hideDialog();
  }

  showDialog(
    type: string,
    label: string,
    user_id: number,
    lead_status: string
  ) {
    this.type = type;
    this.label = label;
    this.user_id = user_id;
    this.lead_status = lead_status;
    this.display = true;
  }

  hideDialog() {
    this.display = false;
    this.lead_status = '';
    this.user_id = null;
  }
}
