import { FdInvestmentDetailComponent } from './components/fd-investment-detail/fd-investment-detail.component';
import { FdInvestmentsListComponent } from './components/fd-investments-list/fd-investments-list.component';
import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
const routes: Routes = [
  { path: '', redirectTo: 'list', pathMatch: 'full' },
  { path: 'list', component: FdInvestmentsListComponent, pathMatch: 'full' },
  {
    path: 'list/:id',
    component: FdInvestmentDetailComponent,
    pathMatch: 'full',
  },
];

@NgModule({
  imports: [RouterModule.forChild(routes)],
  exports: [RouterModule],
})
export class FdInvestmentsRoutingModule {}
