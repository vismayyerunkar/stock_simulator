import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';

import { AssetComponent } from './asset.component';
import { PrimeNgModule } from 'src/app/shared/modules/prime-ng.module';
import { HeaderModule } from '../header/header.module';
import { AssetRoutingModule } from './asset-routing.module';
import { DetailsComponent } from './components/details/details.component';
import { IncomeComponent } from './components/income/income.component';
import { InvestmentsComponent } from './components/investments/investments.component';
import { NotesComponent } from './components/notes/notes.component';
import { SharedModule } from './../../shared/modules/shared.module';
import { AssetListComponent } from './components/asset-list/asset-list.component';
import { AssetDetailComponent } from './components/asset-detail/asset-detail.component';
import { AddAssetComponent } from './components/add-asset/add-asset.component';
import { HttpClientModule } from '@angular/common/http';

@NgModule({
  declarations: [
    AssetComponent,
    DetailsComponent,
    IncomeComponent,
    InvestmentsComponent,
    NotesComponent,
    AssetListComponent,
    AssetDetailComponent,
    AddAssetComponent,
  ],
  imports: [
    CommonModule,
    HeaderModule,
    AssetRoutingModule,
    PrimeNgModule,
    SharedModule,
    HttpClientModule
    
  ],
})
export class AssetModule {}
