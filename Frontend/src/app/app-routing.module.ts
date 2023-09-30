import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { AuthGuard } from 'src/services/auth.guard';
import { NoAuthGuard } from 'src/services/no-auth.guard';
import { StockDetailsComponent } from './pages/stock-details/stock-details.component';
import { WalletComponent } from './pages/wallet/wallet.component';
import { FooterComponent } from './pages/footer/footer.component';
import { WatchlistComponent } from './pages/watchlist/watchlist.component';
import { TestComponent } from './test/test.component';
import { PortfolioComponent } from './portfolio/portfolio.component';
import { ProfileComponent } from './pages/profile/profile.component';
import { SignupComponent } from './pages/signup/signup.component';
import { GoalComponent } from './pages/goal/goal.component';
import { GptComponent } from './pages/gpt/gpt.component';
import { GoalReportComponent } from './pages/goal/components/goal-report/goal-report.component';
import { StocksComponent } from './pages/dashboard/components/stocks/stocks.component';
import { CryptoComponent } from './pages/dashboard/components/crypto/crypto.component';
const routes: Routes = [
  {
    path: 'login',
    loadChildren: () =>
      import('./pages/login/login.module').then((m) => m.LoginModule),
    canActivate: [NoAuthGuard],
  },

  {
    path: 'signup',
    component: SignupComponent,
    pathMatch: 'full',
    canActivate: [NoAuthGuard],
  },
  {
    path: 'stock-details',
    component: StockDetailsComponent,
    pathMatch: 'full',
    canActivate: [AuthGuard],
  },
  {
    path: 'goal-report',
    component: GoalReportComponent,
    pathMatch: 'full',
    canActivate: [AuthGuard],
  },
  {
    path: 'stocks',
    component: StocksComponent,
    pathMatch: 'full',
    canActivate: [AuthGuard],
  },
  {
    path: 'crypto',
    component: CryptoComponent,
    pathMatch: 'full',
    canActivate: [AuthGuard],
  },
  {
    path: 'wallet',
    component: WalletComponent,
    pathMatch: 'full',
    canActivate: [AuthGuard],
  },
  {
    path: 'gpt',
    component: GptComponent,
    pathMatch: 'full',
    canActivate: [AuthGuard],
  },

  {
    path: 'footer',
    component: FooterComponent,
    pathMatch: 'full',
    canActivate: [AuthGuard],
  },
  {
    path: 'watchlist',
    component: WatchlistComponent,
    pathMatch: 'full',
    canActivate: [AuthGuard],
  },
  {
    path: 'test',
    component: TestComponent,
    pathMatch: 'full',
    canActivate: [AuthGuard],
  },
  {
    path: 'portfolio',
    component: PortfolioComponent,
    pathMatch: 'full',
    canActivate: [AuthGuard],
  },
  {
    path: 'profile',
    component: ProfileComponent,
    pathMatch: 'full',
    canActivate: [AuthGuard],
  },
  {
    path: 'goal',
    component: GoalComponent,
    pathMatch: 'full',
    canActivate: [AuthGuard],
  },
  {
    path: 'dashboard',
    loadChildren: () =>
      import('./pages/dashboard/dashboard.module').then(
        (m) => m.DashboardModule
      ),
    canActivate: [AuthGuard],
  },
  {
    path: 'assets',
    loadChildren: () =>
      import('./pages/asset/asset.module').then((m) => m.AssetModule),
    canActivate: [AuthGuard],
    canActivateChild: [AuthGuard],
  },
  {
    path: 'feedback',
    loadChildren: () =>
      import('./pages/feedback/feedback.module').then((m) => m.FeedbackModule),
    canActivate: [AuthGuard],
    canActivateChild: [AuthGuard],
  },
  {
    path: 'support-requests',
    loadChildren: () =>
      import('./pages/supports-request/supports-request.module').then(
        (m) => m.SupportsRequestModule
      ),
    canActivate: [AuthGuard],
    canActivateChild: [AuthGuard],
  },
  {
    path: 'sip-investments',
    loadChildren: () =>
      import('./pages/sip-investments/sip-investments.module').then(
        (m) => m.SipInvestmentsModule
      ),
    canActivate: [AuthGuard],
    canActivateChild: [AuthGuard],
  },
  {
    path: 'fd-investments',
    loadChildren: () =>
      import('./pages/fd-investments/fd-investments.module').then(
        (m) => m.FdInvestmentsModule
      ),
    canActivate: [AuthGuard],
    canActivateChild: [AuthGuard],
  },
  {
    path: 'withdraw',
    loadChildren: () =>
      import('./pages/withdraw-request/withdraw-request.module').then(
        (m) => m.WithdrawRequestModule
      ),
    canActivate: [AuthGuard],
    canActivateChild: [AuthGuard],
  },
  { path: '**', redirectTo: 'dashboard' },
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule],
})
export class AppRoutingModule {}
