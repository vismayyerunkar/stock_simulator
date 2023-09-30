import { PrimeNgModule } from 'src/app/shared/modules/prime-ng.module';
import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { HeaderModule } from '../header/header.module';

import { SupportsRequestRoutingModule } from './supports-request-routing.module';
import { SupportsRequestComponent } from './supports-request.component';
import { SupportsRequestListComponent } from './components/supports-request-list/supports-request-list.component';
import { SupportsRequestDetailsComponent } from './components/supports-request-details/supports-request-details.component';
import { SharedModule } from 'src/app/shared/modules/shared.module';

@NgModule({
  declarations: [
    SupportsRequestComponent,
    SupportsRequestListComponent,
    SupportsRequestDetailsComponent,
  ],
  imports: [
    CommonModule,
    SupportsRequestRoutingModule,
    HeaderModule,
    SharedModule,
    PrimeNgModule,
  ],
})
export class SupportsRequestModule {}
