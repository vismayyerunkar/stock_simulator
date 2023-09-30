import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { PrimeNgModule } from 'src/app/shared/modules/prime-ng.module';
import { HeaderComponent } from './header.component';
import { MenuBarComponent } from './menu-bar/menu-bar.component';
import { SideBarComponent } from './side-bar/side-bar.component';
import { MegaMenuModule } from 'primeng/megamenu';

@NgModule({
  declarations: [HeaderComponent, MenuBarComponent, SideBarComponent],
  imports: [CommonModule, PrimeNgModule, MegaMenuModule],
  exports: [HeaderComponent],
})
export class HeaderModule {}
