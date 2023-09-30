import { Component, OnInit } from '@angular/core';

@Component({
  selector: 'app-gpt',
  templateUrl: './gpt.component.html',
  styleUrls: ['./gpt.component.scss'],
})
export class GptComponent implements OnInit {
  constructor() {}

  ngOnInit(): void {}
  messages: { text: string; isUser: boolean }[] = [];
  newMessage: string = '';
  value: number | null = null;

  sendMessage() {
    if (this.newMessage.trim() !== '') {
      this.messages.push({ text: this.newMessage, isUser: true });
      setTimeout(() => {
        this.messages.push({ text: 'AI Response...', isUser: false });
      }, 1000);

      this.newMessage = '';
    }
  }
}
