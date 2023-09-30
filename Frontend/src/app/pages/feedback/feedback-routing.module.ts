import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { AddFeedbackComponent } from './components/add-feedback/add-feedback.component';
import { FeedbackListComponent } from './components/feedback-list/feedback-list.component';

const routes: Routes = [
  { path: '', redirectTo: 'list', pathMatch: 'full' },
  { path: 'list', component: FeedbackListComponent, pathMatch: 'full' },
  { path: 'add', component: AddFeedbackComponent, pathMatch: 'full' },
];

@NgModule({
  imports: [RouterModule.forChild(routes)],
  exports: [RouterModule]
})
export class FeedbackRoutingModule { }
