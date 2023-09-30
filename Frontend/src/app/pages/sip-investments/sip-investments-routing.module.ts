import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { SipInvestmentsListComponent } from './components/sip-investments-list/sip-investments-list.component';
const routes: Routes = [
  { path: '', redirectTo: 'list', pathMatch: 'full' },
  { path: 'list', component: SipInvestmentsListComponent, pathMatch: 'full' },
];

@NgModule({
  imports: [RouterModule.forChild(routes)],
  exports: [RouterModule],
})
export class SipInvestmentsRoutingModule {}
