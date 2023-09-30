import { SipInvestmentsRoutingModule } from './sip-investments-routing.module';
import { SipInvestmentsListComponent } from './components/sip-investments-list/sip-investments-list.component';
import { SipInvestmentsComponent } from './sip-investments.component';

import { PrimeNgModule } from 'src/app/shared/modules/prime-ng.module';
import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { HeaderModule } from '../header/header.module';
import { SharedModule } from 'src/app/shared/modules/shared.module';

@NgModule({
  declarations: [SipInvestmentsComponent, SipInvestmentsListComponent],
  imports: [
    CommonModule,
    HeaderModule,
    SharedModule,
    PrimeNgModule,
    SipInvestmentsRoutingModule,
  ],
})
export class SipInvestmentsModule {}
