import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { HttpClientModule } from '@angular/common/http';
import { FormsModule, ReactiveFormsModule } from '@angular/forms';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';
import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { DatePipe } from '@angular/common';
import { StockDetailsComponent } from './pages/stock-details/stock-details.component';
import { HeaderModule } from './pages/header/header.module';
import { PanelMenuModule } from 'primeng/panelmenu';
import { CardModule } from 'primeng/card';
import { WalletComponent } from './pages/wallet/wallet.component';
import { ButtonModule } from 'primeng/button';
import { TableModule } from 'primeng/table';
import { FooterComponent } from './pages/footer/footer.component';
import { TabViewModule } from 'primeng/tabview';
import { TabMenuModule } from 'primeng/tabmenu';
import { MenubarModule } from 'primeng/menubar';
import { WalletPieChartComponent } from './pages/wallet-pie-chart/wallet-pie-chart.component';
import { ChartModule } from 'primeng/chart';
import { TagModule } from 'primeng/tag';
import { DataViewModule } from 'primeng/dataview';
import { ProductService } from 'src/services/productservice';
import { TestComponent } from './test/test.component';
import { PortfolioComponent } from './portfolio/portfolio.component';
import { WatchlistModule } from './pages/watchlist/watchlist.module';
import { ProfileComponent } from './pages/profile/profile.component';
import { ImageModule } from 'primeng/image';
import { ToastModule } from 'primeng/toast';
import { SignupComponent } from './pages/signup/signup.component';
import { InputTextModule } from 'primeng/inputtext';
import { SocketIoModule, SocketIoConfig } from 'ngx-socket-io';
import { environment } from 'src/environments/environment';
import { InputNumberModule } from 'primeng/inputnumber';
import { CalendarModule } from 'primeng/calendar';
import { InputTextareaModule } from 'primeng/inputtextarea';
import { InvestedValueComponent } from './portfolio/components/invested-value/invested-value.component';
import { AllComponent } from './pages/wallet/components/all/all.component';
import { StocksComponent } from './pages/wallet/components/stocks/stocks.component';
import { CryptoComponent } from './pages/wallet/components/crypto/crypto.component';
import { GoalComponent } from './pages/goal/goal.component';
import { GptComponent } from './pages/gpt/gpt.component';
import { ToggleButtonModule } from 'primeng/togglebutton';
import { DropdownModule } from 'primeng/dropdown';
import { GoalReportComponent } from './pages/goal/components/goal-report/goal-report.component';
import { ChipsModule } from 'primeng/chips';
import { GoalAssestsComponent } from './pages/goal/components/goal-assests/goal-assests.component';
const config: SocketIoConfig = {
  url: environment?.socketUrl, // socket server url;
  options: {
    transports: ['websocket'],
  },
};
@NgModule({
  declarations: [
    AppComponent,
    StockDetailsComponent,
    WalletComponent,
    FooterComponent,
    WalletPieChartComponent,
    TestComponent,
    PortfolioComponent,
    ProfileComponent,
    SignupComponent,
    InvestedValueComponent,
    AllComponent,
    StocksComponent,
    CryptoComponent,
    GoalComponent,
    GptComponent,
    GoalReportComponent,
    GoalAssestsComponent
  ],
  bootstrap: [AppComponent],
  providers: [DatePipe, ProductService],
  imports: [
    BrowserModule,
    AppRoutingModule,
    FormsModule,
    HttpClientModule,
    BrowserAnimationsModule,
    HeaderModule,
    PanelMenuModule,
    ChipsModule,
    ButtonModule,
    CardModule,
    TableModule,
    TabViewModule,
    TabMenuModule,
    MenubarModule,
    ChartModule,
    ToggleButtonModule,
    InputNumberModule,
    ImageModule,
    TagModule,
    DataViewModule,
    WatchlistModule,
    ReactiveFormsModule,
    DropdownModule,
    ToastModule,
    ButtonModule,
    InputTextModule,
    CalendarModule,
    InputTextareaModule,
    SocketIoModule.forRoot(config),
  ],
})
export class AppModule {}
