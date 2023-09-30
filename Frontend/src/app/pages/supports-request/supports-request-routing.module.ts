import { SupportsRequestListComponent } from './components/supports-request-list/supports-request-list.component';
import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { SupportsRequestDetailsComponent } from './components/supports-request-details/supports-request-details.component';

const routes: Routes = [
  { path: '', redirectTo: 'list', pathMatch: 'full' },
  { path: 'list', component: SupportsRequestListComponent, pathMatch: 'full' },
  {
    path: 'list/:id',
    component: SupportsRequestDetailsComponent,
    pathMatch: 'full',
  },
];

@NgModule({
  imports: [RouterModule.forChild(routes)],
  exports: [RouterModule],
})
export class SupportsRequestRoutingModule { }
