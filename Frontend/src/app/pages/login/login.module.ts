import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';

import { LoginRoutingModule } from './login-routing.module';
import { LoginComponent } from './login.component';
import { PrimeNgModule } from 'src/app/shared/modules/prime-ng.module';
import { SharedModule } from 'src/app/shared/modules/shared.module';


@NgModule({
  declarations: [
    LoginComponent
  ],
  imports: [
    CommonModule,
    LoginRoutingModule,
    PrimeNgModule,
    SharedModule
  ]
})
export class LoginModule { }
