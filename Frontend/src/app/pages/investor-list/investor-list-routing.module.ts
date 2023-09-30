import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { InvestorListComponent } from './investor-list.component';

const routes: Routes = [
  { path: '', component: InvestorListComponent, pathMatch: 'full' },
];

@NgModule({
  imports: [RouterModule.forChild(routes)],
  exports: [RouterModule]
})
export class InvestorListRoutingModule { }
