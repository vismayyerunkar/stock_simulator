import { Component, OnInit, ViewChild } from '@angular/core';
import { AddNoteModalComponent } from 'src/app/shared/components/add-note-modal/add-note-modal.component';
import { Note } from 'src/models/note';

@Component({
  selector: 'app-notes',
  templateUrl: './notes.component.html',
  styleUrls: ['./notes.component.scss'],
})
export class NotesComponent implements OnInit {
  notes: Array<Note> = [];
  @ViewChild(AddNoteModalComponent)
  addNoteModal: AddNoteModalComponent = new AddNoteModalComponent();

  constructor() {}

  ngOnInit(): void {
    for (let i = 1; i <= 3; i++) {
      this.notes.push({
        id: i,
        note: 'Sample Note',
        user: 'Shikhar Agrawal',
        created_on: new Date(),
      });
    }
  }

  addNote(note: string) {
    this.notes.unshift({
      id: this.notes.length + 1,
      note: note,
      user: 'Shikhar Agrawal',
      created_on: new Date(),
    });
  }

  showDialog() {
    this.addNoteModal.showDialog();
  }
}
