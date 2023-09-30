import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { WatchlistComponent } from './watchlist.component';
import { PrimeNgModule } from 'src/app/shared/modules/prime-ng.module';
import { HeaderModule } from '../header/header.module';
import { TabViewModule } from 'primeng/tabview';

@NgModule({
  declarations: [WatchlistComponent],
  exports: [WatchlistComponent],
  imports: [CommonModule, PrimeNgModule, HeaderModule, TabViewModule],
})
export class WatchlistModule {}
