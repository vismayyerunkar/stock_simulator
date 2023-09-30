import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';

import { InvestorListRoutingModule } from './investor-list-routing.module';
import { InvestorListComponent } from './investor-list.component';
import { PrimeNgModule } from 'src/app/shared/modules/prime-ng.module';
import { HeaderModule } from '../header/header.module';
import { SharedModule } from '../../shared/modules/shared.module';

@NgModule({
  declarations: [InvestorListComponent],
  imports: [
    CommonModule,
    HeaderModule,
    InvestorListRoutingModule,
    PrimeNgModule,
    SharedModule,
  ],
})
export class InvestorListModule {}
