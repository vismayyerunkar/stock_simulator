import { PrimeNgModule } from 'src/app/shared/modules/prime-ng.module';
import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { HeaderModule } from '../header/header.module';
import { SharedModule } from 'src/app/shared/modules/shared.module';

import { WithdrawRequestListComponent } from './components/withdraw-request-list/withdraw-request-list.component';
import { WithdrawRequestComponent } from './withdraw-request.component';
import { WithdrawRequestRoutingModule } from './withdraw-request-routing.module';
@NgModule({
  declarations: [WithdrawRequestComponent, WithdrawRequestListComponent],
  imports: [
    CommonModule,
    HeaderModule,
    PrimeNgModule,
    SharedModule,
    WithdrawRequestRoutingModule,
  ],
})
export class WithdrawRequestModule {}
