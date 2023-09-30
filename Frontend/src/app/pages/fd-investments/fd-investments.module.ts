import { FdInvestmentsComponent } from './fd-investments.component';
import { FdInvestmentsListComponent } from './components/fd-investments-list/fd-investments-list.component';
import { PrimeNgModule } from 'src/app/shared/modules/prime-ng.module';
import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { HeaderModule } from '../header/header.module';
import { SharedModule } from 'src/app/shared/modules/shared.module';
import { FdInvestmentsRoutingModule } from './fd-investments-routing.module';
import { FdInvestmentDetailComponent } from './components/fd-investment-detail/fd-investment-detail.component';
import { FdInvestmentUtilizationComponent } from './components/fd-investment-utilization/fd-investment-utilization.component';
import { InvestableAssetListComponent } from './components/investable-asset-list/investable-asset-list.component';

@NgModule({
  declarations: [FdInvestmentsComponent, FdInvestmentsListComponent, FdInvestmentDetailComponent, FdInvestmentUtilizationComponent, InvestableAssetListComponent],
  imports: [
    CommonModule,
    HeaderModule,
    SharedModule,
    PrimeNgModule,
    FdInvestmentsRoutingModule,
  ],
})
export class FdInvestmentsModule {}
