import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { PrimeNgModule } from 'src/app/shared/modules/prime-ng.module';

import { FeedbackRoutingModule } from './feedback-routing.module';
import { FeedbackListComponent } from './components/feedback-list/feedback-list.component';
import { AddFeedbackComponent } from './components/add-feedback/add-feedback.component';
import { FeedbackComponent } from './feedback.component';
import { HeaderModule } from '../header/header.module';

@NgModule({
  declarations: [
    FeedbackListComponent,
    AddFeedbackComponent,
    FeedbackComponent,
  ],
  imports: [CommonModule, FeedbackRoutingModule, HeaderModule, PrimeNgModule],
})
export class FeedbackModule {}
