import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';

import { DashboardRoutingModule } from './dashboard-routing.module';
import { DashboardComponent } from './dashboard.component';
import { HeaderModule } from '../header/header.module';
import { TabViewModule } from 'primeng/tabview';
import { WatchlistModule } from '../watchlist/watchlist.module';
import { StocksComponent } from './components/stocks/stocks.component';
import { CryptoComponent } from './components/crypto/crypto.component';

@NgModule({
  declarations: [DashboardComponent, StocksComponent, CryptoComponent],
  imports: [
    CommonModule,
    HeaderModule,
    DashboardRoutingModule,
    TabViewModule,
    WatchlistModule,
  ],
})
export class DashboardModule {}
